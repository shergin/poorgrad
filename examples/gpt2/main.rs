//! Generates text with OpenAI's released GPT-2 (124M) weights — the
//! whole model recorded on the tape from the existing op surface.
//!
//! Twelve pre-norm blocks: `LayerNorm` into per-head rank-2 attention
//! under a causal mask leaf, `concat` joining the heads, and a GELU
//! MLP (the tanh approximation the checkpoint was trained with, held
//! as scalar leaves — float constants are caller territory). The
//! embedded token window arrives as a per-run input (the vocabulary
//! lookup is loop-land data preparation, a row copy), positions by
//! `narrow`, and the tied language-model head is the embedding table
//! transposed. One forward-only plan at a fixed context serves every
//! generation step, so generating never regrows the tape.
//!
//! The same plan powers two engines. `tape` (the default) runs it on
//! poorgrad's own interpreter. `xla` emits it as StableHLO and holds
//! a serving process (`tools/serve-stablehlo-xla.py`) that compiles
//! it once, keeps the 124M parameters resident, and answers each
//! step over binary pipes — the parameters cross the boundary once,
//! each step ships only the embedded window. `POORGRAD_XLA_PYTHON`
//! names the Python (any with `jax`; default `python3`), and
//! `JAX_PLATFORMS` picks the XLA backend. Stated as measured: XLA-CPU
//! serves at 132 ms/token against the tape's 194 and reproduces its
//! text; `JAX_PLATFORMS=METAL` under a `jax-metal` environment runs
//! at 26 ms/token but miscomputes this module (Apple's experimental
//! plugin; the small conformance modules pass, this one does not) —
//! caught precisely because the tape, XLA-CPU, and the reference
//! interpreter agree with each other.
//!
//! The checkpoint (548 MB) and tokenizer download and cache on first
//! run. Run with:
//! `cargo run --release --features accelerate --example gpt2 -- "prompt" 40 xla`

mod tokenizer;
mod weights;

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use poorgrad::{Differentiable, LayerNorm, Network, Plan, Symbol, Tensor, Value, concat};

use tokenizer::Tokenizer;
use weights::{Weights, cached_text};

/// How many tokens of context the recorded graph attends over.
const CONTEXT_LEN: usize = 256;

/// How many dimensions the residual stream has.
const EMBED_DIM: usize = 768;

/// How many attention heads split the stream.
const HEAD_COUNT: usize = 12;

/// How many dimensions each head reads and writes.
const HEAD_DIM: usize = EMBED_DIM / HEAD_COUNT;

/// How many transformer blocks the model stacks.
const LAYER_COUNT: usize = 12;

/// How many tokens the vocabulary holds.
const VOCABULARY_LEN: usize = 50257;

/// The end-of-text token that opens and closes generation.
const END_OF_TEXT: usize = 50256;

/// The recording-ordered parameter payloads, mirrored as the emitted
/// module's leading arguments.
type Recorder = Vec<Tensor<f32>>;

/// Registers `tensor` as a parameter and records its payload in the
/// module-argument order.
fn parameter<'network>(
    network: &'network Network<Tensor<f32>>,
    recorder: &mut Recorder,
    tensor: Tensor<f32>,
) -> Value<'network, Tensor<f32>> {
    recorder.push(tensor.clone());
    network.parameter(tensor)
}

/// Builds a `LayerNorm`, recording its scale and shift in the
/// module-argument order the facade registers them.
fn layer_norm(
    network: &Network<Tensor<f32>>,
    recorder: &mut Recorder,
    scale: Tensor<f32>,
    shift: Tensor<f32>,
) -> LayerNorm<Tensor<f32>> {
    recorder.push(scale.clone());
    recorder.push(shift.clone());
    LayerNorm::new(network, scale, shift, Tensor::filled([], 1e-5_f32))
}

/// The GELU tanh approximation's constants as scalar leaves, shared
/// by every block.
struct Gelu<'network> {
    half: Value<'network, Tensor<f32>>,
    one: Value<'network, Tensor<f32>>,
    root: Value<'network, Tensor<f32>>,
    coefficient: Value<'network, Tensor<f32>>,
}

impl<'network> Gelu<'network> {
    fn new(network: &'network Network<Tensor<f32>>) -> Self {
        Self {
            half: network.leaf(Tensor::filled([], 0.5_f32)),
            one: network.leaf(Tensor::filled([], 1.0_f32)),
            // The square root of 2 over pi, as the checkpoint's
            // training defined it.
            root: network.leaf(Tensor::filled([], 0.797_884_6_f32)),
            coefficient: network.leaf(Tensor::filled([], 0.044_715_f32)),
        }
    }

    /// Records the tanh-approximated GELU of `x`:
    /// `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
    fn express(&self, x: Value<'network, Tensor<f32>>) -> Value<'network, Tensor<f32>> {
        let cubic = x * x * x * self.coefficient.broadcast_like(x);
        let inner = ((x + cubic) * self.root.broadcast_like(x)).tanh();
        x * (inner + self.one.broadcast_like(inner)) * self.half.broadcast_like(x)
    }
}

/// One recorded block's parameters, loaded from the checkpoint.
struct Block<'network> {
    attention_norm: LayerNorm<Tensor<f32>>,
    attention_weights: Value<'network, Tensor<f32>>,
    attention_bias: Value<'network, Tensor<f32>>,
    projection_weights: Value<'network, Tensor<f32>>,
    projection_bias: Value<'network, Tensor<f32>>,
    hidden_norm: LayerNorm<Tensor<f32>>,
    up_weights: Value<'network, Tensor<f32>>,
    up_bias: Value<'network, Tensor<f32>>,
    down_weights: Value<'network, Tensor<f32>>,
    down_bias: Value<'network, Tensor<f32>>,
}

impl<'network> Block<'network> {
    fn new(
        network: &'network Network<Tensor<f32>>,
        recorder: &mut Recorder,
        weights: &Weights,
        layer: usize,
    ) -> Self {
        let tensor = |suffix: &str| weights.tensor(&format!("h.{layer}.{suffix}"));
        Self {
            attention_norm: layer_norm(
                network,
                recorder,
                tensor("ln_1.weight"),
                tensor("ln_1.bias"),
            ),
            attention_weights: parameter(network, recorder, tensor("attn.c_attn.weight")),
            attention_bias: parameter(network, recorder, tensor("attn.c_attn.bias")),
            projection_weights: parameter(network, recorder, tensor("attn.c_proj.weight")),
            projection_bias: parameter(network, recorder, tensor("attn.c_proj.bias")),
            hidden_norm: layer_norm(
                network,
                recorder,
                tensor("ln_2.weight"),
                tensor("ln_2.bias"),
            ),
            up_weights: parameter(network, recorder, tensor("mlp.c_fc.weight")),
            up_bias: parameter(network, recorder, tensor("mlp.c_fc.bias")),
            down_weights: parameter(network, recorder, tensor("mlp.c_proj.weight")),
            down_bias: parameter(network, recorder, tensor("mlp.c_proj.bias")),
        }
    }

    /// Records the block over the residual stream.
    fn express(
        &self,
        network: &'network Network<Tensor<f32>>,
        stream: Value<'network, Tensor<f32>>,
        mask: Value<'network, Tensor<f32>>,
        scale: Value<'network, Tensor<f32>>,
        gelu: &Gelu<'network>,
    ) -> Value<'network, Tensor<f32>> {
        // Attention reads the normalized stream; every head is a
        // rank-2 slice of one fused query-key-value projection.
        let normalized = self.attention_norm.express(network, stream);
        let fused = normalized.matmul(self.attention_weights);
        let fused = fused + self.attention_bias.broadcast_along(0, fused);
        let heads: Vec<Value<'network, Tensor<f32>>> = (0..HEAD_COUNT)
            .map(|head| {
                let query = fused.narrow(1, head * HEAD_DIM, HEAD_DIM);
                let key = fused.narrow(1, EMBED_DIM + head * HEAD_DIM, HEAD_DIM);
                let value = fused.narrow(1, 2 * EMBED_DIM + head * HEAD_DIM, HEAD_DIM);
                let scores = query.matmul(key.transpose());
                let weights = (scores * scale.broadcast_like(scores) + mask).softmax(1);
                weights.matmul(value)
            })
            .collect();
        let attended = concat(&heads, 1).matmul(self.projection_weights);
        let stream = stream + attended + self.projection_bias.broadcast_along(0, attended);

        // The MLP reads its own normalization of the updated stream.
        let normalized = self.hidden_norm.express(network, stream);
        let up = normalized.matmul(self.up_weights);
        let hidden = gelu.express(up + self.up_bias.broadcast_along(0, up));
        let down = hidden.matmul(self.down_weights);
        stream + down + self.down_bias.broadcast_along(0, down)
    }
}

/// The compiled model: the sampling plan, its feed symbols, and the
/// module-argument payloads.
struct Model {
    plan: Plan<Tensor<f32>>,
    stream: Symbol,
    extraction: Symbol,
    logits: Symbol,
    arguments: Recorder,
}

/// Records GPT-2 from the checkpoint and compiles the sampling plan.
fn model(network: &Network<Tensor<f32>>, weights: &Weights) -> Model {
    let mut recorder = Recorder::new();
    let embeddings = parameter(network, &mut recorder, weights.tensor("wte.weight"));
    let positions = parameter(network, &mut recorder, weights.tensor("wpe.weight"));
    let final_norm = layer_norm(
        network,
        &mut recorder,
        weights.tensor("ln_f.weight"),
        weights.tensor("ln_f.bias"),
    );
    let gelu = Gelu::new(network);
    let scale = network.leaf(Tensor::filled([], 1.0 / (HEAD_DIM as f32).sqrt()));
    let mask_elements: Vec<f32> = (0..CONTEXT_LEN * CONTEXT_LEN)
        .map(|at| {
            if at % CONTEXT_LEN <= at / CONTEXT_LEN {
                0.0
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect();
    let mask = network.leaf(Tensor::new([CONTEXT_LEN, CONTEXT_LEN], mask_elements));
    let blocks: Vec<Block> = (0..LAYER_COUNT)
        .map(|layer| Block::new(network, &mut recorder, weights, layer))
        .collect();

    let embedded = network.input(Tensor::filled([CONTEXT_LEN, EMBED_DIM], 0.0_f32));
    let extraction = network.input(Tensor::selection(vec![0], CONTEXT_LEN, 1.0_f32));

    let mut stream = embedded + positions.narrow(0, 0, CONTEXT_LEN);
    for block in &blocks {
        stream = block.express(network, stream, mask, scale, &gelu);
    }
    let last = final_norm.express(network, stream).gather(extraction);
    // The tied head: logits against the transposed embedding table.
    let logits = last.matmul(embeddings.transpose());

    Model {
        plan: network.compile([logits.symbol()], []),
        stream: embedded.symbol(),
        extraction: extraction.symbol(),
        logits: logits.symbol(),
        arguments: recorder,
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
    /// Emits the plan, stages the arguments, and starts the server.
    fn new(model: &Model) -> Self {
        let directory = weights::cache_directory();
        let module_path = directory.join("gpt2-plan.mlir");
        let static_path = directory.join("gpt2-static.bin");
        let manifest_path = directory.join("gpt2-manifest.json");

        std::fs::write(
            &module_path,
            model.plan.emit_stablehlo().expect("the plan emits"),
        )
        .expect("the module writes");
        let mut arguments = Vec::new();
        for tensor in &model.arguments {
            let axes = tensor.shape().axes().to_vec();
            arguments.extend((axes.len() as u32).to_le_bytes());
            for extent in axes {
                arguments.extend((extent as u32).to_le_bytes());
            }
            for element in tensor.to_vec() {
                arguments.extend(element.to_le_bytes());
            }
        }
        std::fs::write(&static_path, arguments).expect("the arguments write");
        std::fs::write(
            &manifest_path,
            format!("{{\"dynamic\": [[{CONTEXT_LEN}, {EMBED_DIM}], [1, {CONTEXT_LEN}]]}}"),
        )
        .expect("the manifest writes");

        let python = std::env::var("POORGRAD_XLA_PYTHON").unwrap_or_else(|_| "python3".to_string());
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

    let loading = Instant::now();
    let tokenizer = Tokenizer::new(&cached_text("vocab.json"), &cached_text("merges.txt"));
    let weights = Weights::load();
    println!(
        "loaded the checkpoint in {:.1}s",
        loading.elapsed().as_secs_f64()
    );

    let mut window = vec![END_OF_TEXT];
    window.extend(tokenizer.encode(&prompt));
    assert!(
        window.len() + count <= CONTEXT_LEN,
        "prompt and generation must fit the {CONTEXT_LEN}-token context"
    );
    assert_eq!(
        tokenizer.decode(&window[1..]),
        prompt,
        "the tokenizer round-trips the prompt"
    );

    let recording = Instant::now();
    let network = Network::new();
    let model = model(&network, &weights);
    println!(
        "recorded {} nodes and compiled the plan in {:.1}s",
        network.len(),
        recording.elapsed().as_secs_f64()
    );

    // The vocabulary lookup is data preparation: the window embeds by
    // row copies from the table, and the plan adds the positions.
    let table = weights.tensor("wte.weight").to_vec();
    let embedded = |window: &[usize]| {
        let mut stream = vec![0.0_f32; CONTEXT_LEN * EMBED_DIM];
        for (row, &token) in window.iter().enumerate() {
            stream[row * EMBED_DIM..(row + 1) * EMBED_DIM]
                .copy_from_slice(&table[token * EMBED_DIM..(token + 1) * EMBED_DIM]);
        }
        stream
    };

    let mut server = match engine.as_str() {
        "tape" => None,
        "xla" => {
            println!("starting the XLA server (compiling the emitted plan) ...");
            let mut server = XlaServer::new(&model);
            // One warmup step absorbs the server's compile, keeping
            // the per-token figure the steady state.
            let extraction = Tensor::selection(vec![0], CONTEXT_LEN, 1.0_f32);
            server.step(&embedded(&window), &extraction.to_vec());
            Some(server)
        }
        other => panic!("unknown engine `{other}`; use `tape` or `xla`"),
    };

    print!("{prompt}");
    let mut state: u64 = 7;
    let generation = Instant::now();
    for _ in 0..count {
        let stream = embedded(&window);
        let extraction = Tensor::selection(vec![window.len() - 1], CONTEXT_LEN, 1.0_f32);
        let logits = match &mut server {
            Some(server) => server.step(&stream, &extraction.to_vec()),
            None => {
                let evaluation = model.plan.forward(
                    &network,
                    [
                        (model.stream, Tensor::new([CONTEXT_LEN, EMBED_DIM], stream)),
                        (model.extraction, extraction),
                    ],
                );
                evaluation.of(network.resolve(model.logits)).to_vec()
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
    let generated = window.len() - 1 - tokenizer.encode(&prompt).len();
    println!();
    println!(
        "generated {generated} tokens on the {engine} engine in {elapsed:.1}s ({:.0} ms/token)",
        elapsed / generated.max(1) as f64 * 1e3
    );
}
