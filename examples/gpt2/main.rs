//! Generates text with OpenAI's released GPT-2 (124M) weights — the
//! whole model a module tree recorded on the tape from the existing
//! op surface.
//!
//! The model lives in `model.rs` as ordinary [`Module`]
//! implementations: twelve pre-norm blocks (structs of `Linear`s and
//! `LayerNorm`s around a custom attention module) stacked in a plain
//! `Vec`, with the GELU tanh approximation's constants held as
//! scalar leaves — float constants are caller territory. The tree's
//! `visit` paths mirror the checkpoint's own tensor names, so loading
//! is one `named_restore` over the safetensors name map instead of a
//! hand-rolled per-tensor loader. The embedded token arrives as a
//! per-run input (the vocabulary lookup is loop-land data
//! preparation, a row copy), and the tied language-model head is the
//! embedding table transposed. Plans at a fixed context serve every
//! generation step, so generating never regrows the tape.
//!
//! One tape records two expressions of the same parameters
//! (`notes/carry.md`): the full-context window and a one-token decode
//! step whose per-layer key and value caches are caller-carried
//! per-run inputs — the [`Carry`], advanced from each run's declared
//! cache outputs. The default `tape` engine generates through the
//! decode plan (prefill is token-by-token through the same plan);
//! `full` keeps the full-context loop for comparison. `bf16` records
//! the identical module tree over `Tensor<Bf16>` — the genericity the
//! module tier promises, with the matmuls accumulating in f32 by the
//! payload's contract. `xla` emits the full-context f32 plan as
//! StableHLO and holds a serving process
//! (`tools/serve-stablehlo-xla.py`) that compiles it once, keeps the
//! 124M parameters resident, and answers each step over binary pipes —
//! the parameters cross the boundary once, each step ships only the
//! embedded window. `TOPOS_XLA_PYTHON` names the Python (any with
//! `jax`; default `python3`), and `JAX_PLATFORMS` picks the XLA
//! backend. Stated as measured, M1 Pro with `accelerate`: the decode
//! plan generates at 18 ms/token f32 and 31 bf16 against the
//! full-context loop's 193 and 341, reproducing its text token for
//! token; XLA-CPU serves the full-context plan at 132 ms/token and
//! reproduces the same text; `JAX_PLATFORMS=METAL` under a
//! `jax-metal` environment runs at 26 ms/token but miscomputes this
//! module (Apple's experimental plugin; the small conformance
//! modules pass, this one does not) — caught precisely because the
//! tape, XLA-CPU, and the reference interpreter agree with each
//! other.
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
    Bf16, Differentiable, Elementary, Emittable, Module, Plan, Request, Run, Symbol, Tape, Tensor,
    checkpoint,
};

use model::{CONTEXT_LEN, EMBED_DIM, Gpt2, VOCABULARY_LEN, load};
use tokenizer::Tokenizer;
use weights::{Weights, cached_text};

/// The end-of-text token that opens and closes generation.
const END_OF_TEXT: usize = 50256;

/// Which executor and which plan run the generation loop.
enum Engine {
    /// The one-token decode plan on topos's own interpreter.
    Decode,
    /// The full-context plan on topos's own interpreter, kept as the
    /// decode loop's comparison baseline.
    Full,
    /// The full-context plan emitted as StableHLO under a serving XLA
    /// process.
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

/// One layer's cache symbols: the carried inputs and the updated
/// outputs the run pairs them with.
struct LayerCache {
    keys_in: Symbol,
    keys_out: Symbol,
    values_in: Symbol,
    values_out: Symbol,
}

/// The recorded decode step's feed and read symbols.
struct Decoder {
    stream: Symbol,
    position: Symbol,
    mask: Symbol,
    logits: Symbol,
    caches: Vec<LayerCache>,
}

/// Records the one-token decode step over `model`: the embedded row,
/// the position one-hot (placing the cache appends and gathering the
/// position row), the mask row, and the per-layer caches are per-run
/// inputs. The logits are the tied head spelled `(wte . row^T)^T`, so
/// no run materializes the transposed table.
fn record_decode<E: Elementary + From<f32> + 'static>(
    tape: &Tape<Tensor<E>>,
    model: &Gpt2<E>,
) -> Decoder {
    let zeros = |shape: [usize; 2]| Tensor::filled(shape, E::from(0.0));
    let stream = tape.input(zeros([1, EMBED_DIM]));
    let position = tape.input(Tensor::selection(vec![0], CONTEXT_LEN, E::from(1.0)));
    let mask = tape.input(zeros([1, CONTEXT_LEN]));
    let pairs: Vec<_> = (0..model.layers())
        .map(|_| {
            (
                tape.input(zeros([CONTEXT_LEN, EMBED_DIM])),
                tape.input(zeros([CONTEXT_LEN, EMBED_DIM])),
            )
        })
        .collect();
    let (last, updated) = model.express_decode(tape, stream, &pairs, position, mask);
    let logits = tape
        .resolve(model.embeddings())
        .matmul(last.transpose())
        .transpose();
    let caches = pairs
        .iter()
        .zip(&updated)
        .map(
            |((keys_in, values_in), (keys_out, values_out))| LayerCache {
                keys_in: keys_in.symbol(),
                keys_out: keys_out.symbol(),
                values_in: values_in.symbol(),
                values_out: values_out.symbol(),
            },
        )
        .collect();
    Decoder {
        stream: stream.symbol(),
        position: position.symbol(),
        mask: mask.symbol(),
        logits: logits.symbol(),
        caches,
    }
}

/// The decode loop's pending feeds: one payload per carried cache
/// input, advanced from each run's declared cache outputs
/// (`notes/carry.md`).
///
/// Feeding and advancing clone the caches; the zero-copy spelling (a
/// consuming read on `Run`, the unfed-only default overlay in
/// `Plan::forward`) was built and graded flat on both axes at this
/// scale — ~57 MB of clones per step ride under the ~500 MB of
/// weight traffic — so the simpler spelling stays, per the numbers.
struct Carry<E> {
    entries: Vec<(Symbol, Tensor<E>)>,
}

impl<E: Elementary + From<f32>> Carry<E> {
    /// Returns the empty caches: generation's initial state.
    fn new(caches: &[LayerCache]) -> Self {
        let zeros = || Tensor::filled([CONTEXT_LEN, EMBED_DIM], E::from(0.0));
        let entries = caches
            .iter()
            .flat_map(|cache| [(cache.keys_in, zeros()), (cache.values_in, zeros())])
            .collect();
        Self { entries }
    }

    /// Returns this step's cache feeds.
    fn feeds(&self) -> impl Iterator<Item = (Symbol, Tensor<E>)> + '_ {
        self.entries
            .iter()
            .map(|(symbol, payload)| (*symbol, payload.clone()))
    }

    /// Returns the carry advanced from `run`: each cache input's next
    /// payload is its pair's output in the run.
    fn advanced(run: &Run<Tensor<E>>, caches: &[LayerCache]) -> Self {
        let entries = caches
            .iter()
            .flat_map(|cache| {
                [
                    (cache.keys_in, run.of(cache.keys_out).clone()),
                    (cache.values_in, run.of(cache.values_out).clone()),
                ]
            })
            .collect();
        Self { entries }
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
    let decoder = record_decode(&tape, &gpt2);
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

    if let Engine::Decode = engine {
        let compiling = Instant::now();
        let outputs: Vec<Symbol> = decoder
            .caches
            .iter()
            .flat_map(|cache| [cache.keys_out, cache.values_out])
            .collect();
        let plan: Plan<Tensor<E>> =
            network.compile(Request::roots([decoder.logits]).observe(outputs));
        println!(
            "recorded {} nodes and compiled the decode plan in {:.1}s",
            network.len(),
            compiling.elapsed().as_secs_f64()
        );

        let table = parameters.of(gpt2.embeddings()).to_vec();
        let row = |token: usize| {
            Tensor::new(
                [1, EMBED_DIM],
                table[token * EMBED_DIM..(token + 1) * EMBED_DIM].to_vec(),
            )
        };
        let mask_row = |until: usize| {
            let elements: Vec<E> = (0..CONTEXT_LEN)
                .map(|at| {
                    if at <= until {
                        E::from(0.0)
                    } else {
                        E::from(f32::NEG_INFINITY)
                    }
                })
                .collect();
            Tensor::new([1, CONTEXT_LEN], elements)
        };
        // One step: feed the carry and the row's transients, read the
        // logits, advance the carry from the run's cache outputs.
        let step = |carry: Carry<E>, token: usize, at: usize| -> (Vec<f32>, Carry<E>) {
            let run = plan.forward(
                &parameters,
                carry.feeds().chain([
                    (decoder.stream, row(token)),
                    (
                        decoder.position,
                        Tensor::selection(vec![at], CONTEXT_LEN, E::from(1.0)),
                    ),
                    (decoder.mask, mask_row(at)),
                ]),
            );
            let logits = run.of(decoder.logits).to_vec();
            (
                logits.iter().map(|&element| f32::from(element)).collect(),
                Carry::advanced(&run, &decoder.caches),
            )
        };

        // Prefill is token-by-token through the same plan: the carry
        // is the only state the steps share.
        let prefilling = Instant::now();
        let mut carry = Carry::new(&decoder.caches);
        let mut logits = Vec::new();
        for (at, &token) in window.iter().enumerate() {
            (logits, carry) = step(carry, token, at);
        }
        println!(
            "prefilled {} tokens in {:.2}s",
            window.len(),
            prefilling.elapsed().as_secs_f64()
        );

        print!("{prompt}");
        let mut state: u64 = 7;
        let generation = Instant::now();
        for index in 0..count {
            let token = draw(&logits, 0.9, 40, &mut state);
            if token == END_OF_TEXT {
                break;
            }
            window.push(token);
            print!("{}", tokenizer.decode(&[token]));
            std::io::stdout().flush().expect("stdout flushes");
            if index + 1 < count {
                (logits, carry) = step(carry, token, window.len() - 1);
            }
        }
        let elapsed = generation.elapsed().as_secs_f64();
        let generated = window.len() - 1 - tokenizer.encode(prompt).len();
        println!();
        println!(
            "generated {generated} tokens on the {label} engine in {elapsed:.1}s ({:.0} ms/token)",
            elapsed / generated.max(1) as f64 * 1e3
        );
        return;
    }

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
        Engine::Decode | Engine::Full => None,
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
        "tape" => run::<f32>(&prompt, count, Engine::Decode, "tape"),
        "full" => run::<f32>(&prompt, count, Engine::Full, "full"),
        "xla" => run::<f32>(&prompt, count, Engine::Xla, "xla"),
        "bf16" => run::<Bf16>(&prompt, count, Engine::Decode, "bf16"),
        other => panic!("unknown engine `{other}`; use `tape`, `full`, `xla`, or `bf16`"),
    }
}
