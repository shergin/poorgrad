//! Generates text with OpenAI's released GPT-2 (124M) weights — the
//! whole model recorded on the tape from the existing op surface.
//!
//! Twelve pre-norm blocks: `LayerNorm` into per-head rank-2 attention
//! under a causal mask leaf, `concat` joining the heads, and a GELU
//! MLP (the tanh approximation the checkpoint was trained with, held
//! as scalar leaves — float constants are caller territory). The
//! token and position embeddings arrive by `gather` and `narrow`, and
//! the tied language-model head is the embedding table transposed.
//! One forward-only plan at a fixed context serves every generation
//! step: the token window and the prediction row's one-hot extraction
//! are per-run inputs, so generating never regrows the tape.
//!
//! The checkpoint (548 MB) and tokenizer download and cache on first
//! run. Run with:
//! `cargo run --release --features accelerate --example gpt2 -- "prompt"`

mod json;
mod tokenizer;
mod weights;

use std::time::Instant;

use poorgrad::{LayerNorm, Network, Plan, Symbol, Tensor, Value, concat};

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
    fn new(network: &'network Network<Tensor<f32>>, weights: &Weights, layer: usize) -> Self {
        let epsilon = Tensor::filled([], 1e-5_f32);
        let tensor = |suffix: &str| weights.tensor(&format!("h.{layer}.{suffix}"));
        Self {
            attention_norm: LayerNorm::new(
                network,
                tensor("ln_1.weight"),
                tensor("ln_1.bias"),
                epsilon.clone(),
            ),
            attention_weights: network.parameter(tensor("attn.c_attn.weight")),
            attention_bias: network.parameter(tensor("attn.c_attn.bias")),
            projection_weights: network.parameter(tensor("attn.c_proj.weight")),
            projection_bias: network.parameter(tensor("attn.c_proj.bias")),
            hidden_norm: LayerNorm::new(
                network,
                tensor("ln_2.weight"),
                tensor("ln_2.bias"),
                epsilon,
            ),
            up_weights: network.parameter(tensor("mlp.c_fc.weight")),
            up_bias: network.parameter(tensor("mlp.c_fc.bias")),
            down_weights: network.parameter(tensor("mlp.c_proj.weight")),
            down_bias: network.parameter(tensor("mlp.c_proj.bias")),
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

/// The compiled model: the sampling plan and its feed symbols.
struct Model {
    plan: Plan<Tensor<f32>>,
    tokens: Symbol,
    extraction: Symbol,
    logits: Symbol,
}

/// Records GPT-2 from the checkpoint and compiles the sampling plan.
fn model(network: &Network<Tensor<f32>>, weights: &Weights) -> Model {
    let embeddings = network.parameter(weights.tensor("wte.weight"));
    let positions = network.parameter(weights.tensor("wpe.weight"));
    let final_norm = LayerNorm::new(
        network,
        weights.tensor("ln_f.weight"),
        weights.tensor("ln_f.bias"),
        Tensor::filled([], 1e-5_f32),
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
        .map(|layer| Block::new(network, weights, layer))
        .collect();

    let tokens = network.input(Tensor::selection(
        vec![END_OF_TEXT; CONTEXT_LEN],
        VOCABULARY_LEN,
        1.0_f32,
    ));
    let extraction = network.input(Tensor::selection(vec![0], CONTEXT_LEN, 1.0_f32));

    let mut stream = embeddings.gather(tokens) + positions.narrow(0, 0, CONTEXT_LEN);
    for block in &blocks {
        stream = block.express(network, stream, mask, scale, &gelu);
    }
    let last = final_norm.express(network, stream).gather(extraction);
    // The tied head: logits against the transposed embedding table.
    let logits = last.matmul(embeddings.transpose());

    Model {
        plan: network.compile([logits.symbol()], []),
        tokens: tokens.symbol(),
        extraction: extraction.symbol(),
        logits: logits.symbol(),
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

    print!("{prompt}");
    let mut state: u64 = 7;
    let generation = Instant::now();
    for _ in 0..count {
        let mut padded = window.clone();
        padded.resize(CONTEXT_LEN, END_OF_TEXT);
        let evaluation = model.plan.forward(
            &network,
            [
                (
                    model.tokens,
                    Tensor::selection(padded, VOCABULARY_LEN, 1.0_f32),
                ),
                (
                    model.extraction,
                    Tensor::selection(vec![window.len() - 1], CONTEXT_LEN, 1.0_f32),
                ),
            ],
        );
        let logits = evaluation.of(network.resolve(model.logits)).to_vec();
        let token = draw(&logits, 0.9, 40, &mut state);
        if token == END_OF_TEXT {
            break;
        }
        window.push(token);
        print!("{}", tokenizer.decode(&[token]));
        use std::io::Write;
        std::io::stdout().flush().expect("stdout flushes");
    }
    let elapsed = generation.elapsed().as_secs_f64();
    let generated = window.len() - 1 - tokenizer.encode(&prompt).len();
    println!();
    println!(
        "generated {generated} tokens in {elapsed:.1}s ({:.0} ms/token)",
        elapsed / generated.max(1) as f64 * 1e3
    );
}
