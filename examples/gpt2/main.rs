//! Generates text with OpenAI's released GPT-2 (124M) weights — the
//! whole model a module tree recorded on the tape from the existing
//! op surface.
//!
//! The model lives in `model.rs` as ordinary [`Module`]
//! implementations: twelve pre-norm blocks (structs of `Linear`s and
//! `LayerNorm`s around a custom attention module) stacked in a
//! `Sequential`, with the GELU tanh approximation's constants held as
//! scalar leaves — float constants are caller territory. The tree's
//! `visit` paths mirror the checkpoint's own tensor names, so loading
//! is one `named_restore` over the safetensors name map instead of a
//! hand-rolled per-tensor loader. The embedded token window arrives
//! as a per-run input (the vocabulary lookup is loop-land data
//! preparation, a row copy), and the tied language-model head is the
//! embedding table transposed. One forward-only plan at a fixed
//! context serves every generation step, so generating never regrows
//! the tape.
//!
//! The same plan powers three runs. `tape` (the default) runs it on
//! topos's own interpreter. `bf16` records the identical module
//! tree over `Tensor<Bf16>` — the genericity the module tier
//! promises, with the matmuls accumulating in f32 by the payload's
//! contract; measured at 341 ms/token against the f32 tape's 195 on
//! the same machine. `xla` emits the f32 plan as
//! StableHLO and holds a serving process
//! (`tools/serve-stablehlo-xla.py`) that compiles it once, keeps the
//! 124M parameters resident, and answers each step over binary pipes —
//! the parameters cross the boundary once, each step ships only the
//! embedded window. `TOPOS_XLA_PYTHON` names the Python (any with
//! `jax`; default `python3`), and `JAX_PLATFORMS` picks the XLA
//! backend. Stated as measured: XLA-CPU serves at 132 ms/token
//! against the tape's 194 and reproduces its text; `JAX_PLATFORMS=METAL`
//! under a `jax-metal` environment runs at 26 ms/token but
//! miscomputes this module (Apple's experimental plugin; the small
//! conformance modules pass, this one does not) — caught precisely
//! because the tape, XLA-CPU, and the reference interpreter agree
//! with each other.
//!
//! The checkpoint (548 MB) and tokenizer download and cache on first
//! run. Run with:
//! `cargo run --release --features accelerate --example gpt2 -- "prompt" 40 xla`

mod model;
mod tokenizer;
mod weights;

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use topos::{
    Bf16, Differentiable, Elementary, Emittable, Module, Plan, Request, Symbol, Tape, Tensor,
    checkpoint,
};

use model::{CONTEXT_LEN, EMBED_DIM, Gpt2, VOCABULARY_LEN, load};
use tokenizer::Tokenizer;
use weights::{Weights, cached_text};

/// The end-of-text token that opens and closes generation.
const END_OF_TEXT: usize = 50256;

/// Which executor runs the compiled plan.
enum Engine {
    /// Topos's own interpreter.
    Tape,
    /// The emitted StableHLO under a serving XLA process.
    Xla,
}

/// The recorded sampling expression's feed and read symbols.
struct Sampler {
    stream: Symbol,
    extraction: Symbol,
    logits: Symbol,
}

/// Records the sampling expression over `model`: the embedded window
/// and the extraction row are per-run inputs, and the logits are the
/// tied head — the embedding table transposed.
fn record<E: Elementary + From<f32> + 'static>(tape: &Tape<Tensor<E>>, model: &Gpt2<E>) -> Sampler {
    let embedded = tape.input(Tensor::filled([CONTEXT_LEN, EMBED_DIM], E::from(0.0)));
    let extraction = tape.input(Tensor::selection(vec![0], CONTEXT_LEN, E::from(1.0)));
    let last = model.express(tape, embedded).gather(extraction);
    let logits = last.matmul(tape.resolve(model.embeddings()).transpose());
    Sampler {
        stream: embedded.symbol(),
        extraction: extraction.symbol(),
        logits: logits.symbol(),
    }
}

/// The XLA serving process: the emitted plan compiled once, the
/// parameters resident, one execution per written step.
struct XlaServer {
    child: Child,
    requests: ChildStdin,
    responses: ChildStdout,
}

impl XlaServer {
    /// Emits the plan, stages `arguments` — the parameter payloads in
    /// the emitted argument order — and starts the server.
    fn new<E>(plan: &Plan<Tensor<E>>, arguments: &[Tensor<E>]) -> Self
    where
        E: Elementary + Emittable + Copy,
        f32: From<E>,
    {
        let directory = weights::cache_directory();
        let module_path = directory.join("gpt2-plan.mlir");
        let static_path = directory.join("gpt2-static.bin");
        let manifest_path = directory.join("gpt2-manifest.json");

        std::fs::write(&module_path, plan.emit_stablehlo().expect("the plan emits"))
            .expect("the module writes");
        let mut staged = Vec::new();
        for tensor in arguments {
            let axes = tensor.shape().axes().to_vec();
            staged.extend((axes.len() as u32).to_le_bytes());
            for extent in axes {
                staged.extend((extent as u32).to_le_bytes());
            }
            for element in tensor.to_vec() {
                staged.extend(f32::from(element).to_le_bytes());
            }
        }
        std::fs::write(&static_path, staged).expect("the arguments write");
        std::fs::write(
            &manifest_path,
            format!("{{\"dynamic\": [[{CONTEXT_LEN}, {EMBED_DIM}], [1, {CONTEXT_LEN}]]}}"),
        )
        .expect("the manifest writes");

        let python = std::env::var("TOPOS_XLA_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let mut command: Vec<String> = python.split_whitespace().map(str::to_string).collect();
        command.push("tools/serve-stablehlo-xla.py".to_string());
        let mut child = Command::new(&command[0])
            .args(&command[1..])
            .arg(&module_path)
            .arg(&static_path)
            .arg(&manifest_path)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()
            .expect("the serving process starts; is `jax` installed for it?");
        let requests = child.stdin.take().expect("the server's input pipes");
        let responses = child.stdout.take().expect("the server's output pipes");
        Self {
            child,
            requests,
            responses,
        }
    }

    /// Executes one step and returns the logits.
    fn step(&mut self, stream: &[f32], extraction: &[f32]) -> Vec<f32> {
        let mut request = Vec::with_capacity(4 * (stream.len() + extraction.len()));
        for &value in stream.iter().chain(extraction) {
            request.extend(value.to_le_bytes());
        }
        self.requests
            .write_all(&request)
            .expect("the request writes");
        self.requests.flush().expect("the request flushes");
        let mut response = vec![0u8; 4 * VOCABULARY_LEN];
        self.responses
            .read_exact(&mut response)
            .expect("the server answers; see its standard error");
        response
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect()
    }
}

impl Drop for XlaServer {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Advances `state` and returns the next value uniformly in `[0, 1)`.
fn unit(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let bits = (*state >> 11) as f64;
    bits / (1u64 << 53) as f64
}

/// Draws one token from `logits` under temperature and top-k.
fn draw(logits: &[f32], temperature: f64, top: usize, state: &mut u64) -> usize {
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(top);
    let peak = ranked[0].1 as f64;
    let weights: Vec<f64> = ranked
        .iter()
        .map(|&(_, logit)| ((logit as f64 - peak) / temperature).exp())
        .collect();
    let total: f64 = weights.iter().sum();
    let mut remaining = unit(state) * total;
    for (&(id, _), weight) in ranked.iter().zip(&weights) {
        if remaining < *weight {
            return id;
        }
        remaining -= weight;
    }
    ranked[0].0
}

/// Loads the checkpoint into a module tree of element type `E`,
/// compiles the sampling plan, and generates `count` tokens after
/// `prompt` on `engine`, reporting timings as `label`.
fn run<E>(prompt: &str, count: usize, engine: Engine, label: &str)
where
    E: Elementary + Emittable + From<f32> + Copy + 'static,
    f32: From<E>,
{
    let loading = Instant::now();
    let tokenizer = Tokenizer::new(&cached_text("vocab.json"), &cached_text("merges.txt"));
    let weights = Weights::load();

    // The module tree and the sampling expression record on one tape;
    // sealing yields the immutable spec, and the named restore builds
    // the parameter state that carries the checkpoint, converting
    // elements at the precision boundary.
    let tape = Tape::new();
    let gpt2 = Gpt2::<E>::new(&tape);
    let sampler = record(&tape, &gpt2);
    let network = tape.into_network();
    let parameters = load(&network.parameters(), &gpt2, &weights);
    drop(weights);
    println!(
        "loaded the checkpoint in {:.1}s",
        loading.elapsed().as_secs_f64()
    );

    let mut window = vec![END_OF_TEXT];
    window.extend(tokenizer.encode(prompt));
    assert!(
        window.len() + count <= CONTEXT_LEN,
        "prompt and generation must fit the {CONTEXT_LEN}-token context"
    );
    assert_eq!(
        tokenizer.decode(&window[1..]),
        prompt,
        "the tokenizer round-trips the prompt"
    );

    let compiling = Instant::now();
    let plan: Plan<Tensor<E>> = network.compile(Request::roots([sampler.logits]));
    println!(
        "recorded {} nodes and compiled the plan in {:.1}s",
        network.len(),
        compiling.elapsed().as_secs_f64()
    );

    // The vocabulary lookup is data preparation: the window embeds by
    // row copies from the table, and the plan adds the positions.
    let table = parameters.of(gpt2.embeddings()).to_vec();
    let embedded = |window: &[usize]| {
        let mut stream = vec![E::from(0.0); CONTEXT_LEN * EMBED_DIM];
        for (row, &token) in window.iter().enumerate() {
            stream[row * EMBED_DIM..(row + 1) * EMBED_DIM]
                .copy_from_slice(&table[token * EMBED_DIM..(token + 1) * EMBED_DIM]);
        }
        stream
    };
    let widened = |elements: &[E]| -> Vec<f32> {
        elements.iter().map(|&element| f32::from(element)).collect()
    };

    let mut server = match engine {
        Engine::Tape => None,
        Engine::Xla => {
            println!("starting the XLA server (compiling the emitted plan) ...");
            // The emitted module's leading arguments are the
            // parameters in recording order; the tree records them in
            // visit order, so the positional snapshot is exactly the
            // argument list.
            let arguments = checkpoint::snapshot(&parameters, &gpt2);
            let mut server = XlaServer::new(&plan, &arguments);
            // One warmup step absorbs the server's compile, keeping
            // the per-token figure the steady state.
            let extraction = Tensor::selection(vec![0], CONTEXT_LEN, 1.0_f32);
            server.step(&widened(&embedded(&window)), &extraction.to_vec());
            Some(server)
        }
    };

    print!("{prompt}");
    let mut state: u64 = 7;
    let generation = Instant::now();
    for _ in 0..count {
        let stream = embedded(&window);
        let extraction = Tensor::selection(vec![window.len() - 1], CONTEXT_LEN, E::from(1.0));
        let logits = match &mut server {
            Some(server) => server.step(&widened(&stream), &widened(&extraction.to_vec())),
            None => {
                let run = plan.forward(
                    &parameters,
                    [
                        (
                            sampler.stream,
                            Tensor::new([CONTEXT_LEN, EMBED_DIM], stream),
                        ),
                        (sampler.extraction, extraction),
                    ],
                );
                widened(&run.of(sampler.logits).to_vec())
            }
        };
        let token = draw(&logits, 0.9, 40, &mut state);
        if token == END_OF_TEXT {
            break;
        }
        window.push(token);
        print!("{}", tokenizer.decode(&[token]));
        std::io::stdout().flush().expect("stdout flushes");
    }
    let elapsed = generation.elapsed().as_secs_f64();
    let generated = window.len() - 1 - tokenizer.encode(prompt).len();
    println!();
    println!(
        "generated {generated} tokens on the {label} engine in {elapsed:.1}s ({:.0} ms/token)",
        elapsed / generated.max(1) as f64 * 1e3
    );
}

fn main() {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The library of the poor holds one book".to_string());
    let count: usize = std::env::args()
        .nth(2)
        .map(|argument| argument.parse().expect("a token count"))
        .unwrap_or(40);
    let engine = std::env::args()
        .nth(3)
        .unwrap_or_else(|| "tape".to_string());

    match engine.as_str() {
        "tape" => run::<f32>(&prompt, count, Engine::Tape, "tape"),
        "xla" => run::<f32>(&prompt, count, Engine::Xla, "xla"),
        "bf16" => run::<Bf16>(&prompt, count, Engine::Tape, "bf16"),
        other => panic!("unknown engine `{other}`; use `tape`, `xla`, or `bf16`"),
    }
}
