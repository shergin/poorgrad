//! Generates text with released Llama-family weights — the whole
//! architecture a module tree recorded on the tape from the existing
//! op surface.
//!
//! The model lives in `model.rs` as ordinary [`Module`]
//! implementations: pre-norm blocks (structs of bias-free projections
//! and `RmsNorm`s around a grouped-query attention module) stacked in
//! a `Sequential`, with rotary position embeddings as precomputed
//! cosine and sine leaves and the SwiGLU MLP spelling SiLU from the
//! op surface — no new opcodes. The architecture is one and a model
//! is data: a `Family` descriptor picks TinyLlama 1.1B (the default)
//! or Llama 2 7B, and the same structs record either. The tree's
//! `visit` paths mirror the checkpoints' own tensor names, so loading
//! is one `named_restore` over tensors streamed shard by shard. The
//! embedded token window arrives as a per-run input (the vocabulary
//! lookup is loop-land data preparation, a row copy), and the untied
//! language-model head records after the prediction row's one-hot
//! extraction. One forward-only plan at a fixed context serves every
//! generation step, so generating never regrows the tape.
//!
//! The same plan powers two runs: `tape` (the default) runs it on
//! topos's own interpreter over f32, and `bf16` records the identical
//! module tree over `Tensor<Bf16>` — the genericity the module tier
//! promises, with the matmuls accumulating in f32 by the payload's
//! contract. At 7B, `bf16` is also the practical engine: it keeps the
//! resident parameters near 13 GB where f32 wants 27.
//!
//! The checkpoint and tokenizer download and cache on first run. Run
//! with:
//! `cargo run --release --features accelerate --example llama -- "prompt" 40 bf16 llama2`

mod family;
mod model;
mod tokenizer;
mod weights;

use std::io::Write;
use std::time::Instant;

use topos::{Bf16, Elementary, Module, Plan, Request, Symbol, Tape, Tensor};

use family::{Family, LLAMA2_7B, TINYLLAMA};
use model::{CONTEXT_LEN, Llama, load};
use tokenizer::Tokenizer;
use weights::{Weights, cached_text};

/// The beginning-of-sequence token that opens generation.
const SEQUENCE_START: usize = 1;

/// The end-of-sequence token that closes generation.
const SEQUENCE_END: usize = 2;

/// The recorded sampling expression's feed and read symbols.
struct Sampler {
    stream: Symbol,
    extraction: Symbol,
    logits: Symbol,
}

/// Records the sampling expression over `model`: the embedded window
/// and the extraction row are per-run inputs, and the logits are the
/// untied head over the extracted row.
fn record<E: Elementary + From<f32> + 'static>(
    tape: &Tape<Tensor<E>>,
    family: Family,
    model: &Llama<E>,
) -> Sampler {
    let embedded = tape.input(Tensor::filled(
        [CONTEXT_LEN, family.embed_dim],
        E::from(0.0),
    ));
    let extraction = tape.input(Tensor::selection(vec![0], CONTEXT_LEN, E::from(1.0)));
    let last = model.express(tape, embedded).gather(extraction);
    let logits = model.predict(tape, last);
    Sampler {
        stream: embedded.symbol(),
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

/// Loads `family`'s checkpoint into a module tree of element type
/// `E`, compiles the sampling plan, and generates `count` tokens
/// after `prompt`, reporting timings as `label`.
fn run<E>(family: Family, prompt: &str, count: usize, label: &str)
where
    E: Elementary + From<f32> + Copy + 'static,
    f32: From<E>,
{
    let loading = Instant::now();
    let tokenizer = Tokenizer::new(&cached_text(&family, "tokenizer.json"));
    let weights = Weights::open(&family);

    // The module tree and the sampling expression record on one tape;
    // sealing yields the immutable spec, and the named restore builds
    // the parameter state that carries the checkpoint, converting
    // elements at the precision boundary.
    let tape = Tape::new();
    let llama = Llama::<E>::new(&tape, family);
    let sampler = record(&tape, family, &llama);
    let network = tape.into_network();
    let parameters = load(&network.parameters(), &llama, &weights);
    drop(weights);
    println!(
        "loaded the checkpoint in {:.1}s",
        loading.elapsed().as_secs_f64()
    );

    let mut window = vec![SEQUENCE_START];
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
    // row copies from the table; position enters inside the plan
    // through the rotary leaves.
    let embed_dim = family.embed_dim;
    let table = parameters.of(llama.embeddings()).to_vec();
    let embedded = |window: &[usize]| {
        let mut stream = vec![E::from(0.0); CONTEXT_LEN * embed_dim];
        for (row, &token) in window.iter().enumerate() {
            stream[row * embed_dim..(row + 1) * embed_dim]
                .copy_from_slice(&table[token * embed_dim..(token + 1) * embed_dim]);
        }
        stream
    };

    print!("{prompt}");
    let mut state: u64 = 7;
    let generation = Instant::now();
    for _ in 0..count {
        let stream = embedded(&window);
        let extraction = Tensor::selection(vec![window.len() - 1], CONTEXT_LEN, E::from(1.0));
        let run = plan.forward(
            &parameters,
            [
                (
                    sampler.stream,
                    Tensor::new([CONTEXT_LEN, embed_dim], stream),
                ),
                (sampler.extraction, extraction),
            ],
        );
        let logits: Vec<f32> = run
            .of(sampler.logits)
            .to_vec()
            .into_iter()
            .map(f32::from)
            .collect();
        let token = draw(&logits, 0.9, 40, &mut state);
        if token == SEQUENCE_END {
            break;
        }
        window.push(token);
        print!("{}", tokenizer.piece(token));
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
    let member = std::env::args()
        .nth(4)
        .unwrap_or_else(|| "tinyllama".to_string());

    let family = match member.as_str() {
        "tinyllama" => TINYLLAMA,
        "llama2" => LLAMA2_7B,
        other => panic!("unknown model `{other}`; use `tinyllama` or `llama2`"),
    };
    match engine.as_str() {
        "tape" => run::<f32>(family, &prompt, count, "tape"),
        "bf16" => run::<Bf16>(family, &prompt, count, "bf16"),
        other => panic!("unknown engine `{other}`; use `tape` or `bf16`"),
    }
}
