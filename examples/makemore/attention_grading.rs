//! Grades the two recordings of the transformer's training step —
//! the rank-2 head loop against batched attention — as *emission
//! source material*: both record the same joint step
//! `(parameters, batch) -> (loss, gradients...)`, both emit to
//! StableHLO, and one resident XLA server times each over repeated
//! identical requests. The in-crate B1b grading declined the batched
//! migration at this scale (`notes/batched-matmul.md`); this harness
//! measures the batched op's remaining execution claim — that
//! `dot_general` with batching dimensions is better source for
//! someone else's compiler than a head loop it must re-fuse.
//!
//! `TOPOS_XLA_PYTHON` names the serving interpreter (default
//! `python3`); the backend follows jax's own selection, so
//! `JAX_PLATFORMS` picks a PJRT plugin — CPU by default, Metal when
//! the `jax-metal` plugin is installed in that interpreter.
//!
//! Run with: `cargo run --release --example makemore_attention_grading`

// The shared corpus module also serves the sampling examples; this
// harness never samples, so most of it stays unused here.
#[allow(dead_code)]
mod corpus;

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use topos::{
    Differentiable, Request, RmsNorm, Shape, Symbol, Tape, Tensor, Value, concat, cross_entropy,
    init, stack,
};

use corpus::{VOCABULARY_LEN, load_names, shuffle, training_samples};

/// How many characters of history the model attends over.
const CONTEXT_LEN: usize = 8;

/// How many dimensions the residual stream has.
const EMBED_DIM: usize = 32;

/// How many attention heads split the stream.
const HEAD_COUNT: usize = 2;

/// How many dimensions each head reads and writes.
const HEAD_DIM: usize = EMBED_DIM / HEAD_COUNT;

/// How many neurons the feed-forward hidden layer has.
const HIDDEN_LEN: usize = 4 * EMBED_DIM;

/// How many samples each training step packs into the attention row.
const BATCH_LEN: usize = 32;

/// How many token rows the packed training batch holds.
const PACKED_LEN: usize = BATCH_LEN * CONTEXT_LEN;

/// How many timed requests each module serves, after warmup.
const TIMED_STEPS: usize = 300;

/// How many warmup requests precede the timing.
const WARMUP_STEPS: usize = 20;

/// The parameter payloads in recording order — the emitted module's
/// leading arguments. Identical to the transformer act's seeds.
fn parameter_payloads() -> Vec<Tensor<f32>> {
    let mut weights = init::xavier(7);
    let ones = Tensor::filled([EMBED_DIM], 1.0);
    let mut payloads = vec![
        init::normal(8, 1.0)(&Shape::new([VOCABULARY_LEN, EMBED_DIM])),
        init::normal(9, 1.0)(&Shape::new([CONTEXT_LEN, EMBED_DIM])),
    ];
    for _ in 0..HEAD_COUNT {
        payloads.push(weights(&Shape::new([EMBED_DIM, HEAD_DIM])));
        payloads.push(weights(&Shape::new([EMBED_DIM, HEAD_DIM])));
        payloads.push(weights(&Shape::new([EMBED_DIM, HEAD_DIM])));
    }
    payloads.push(weights(&Shape::new([EMBED_DIM, EMBED_DIM])));
    payloads.push(ones.clone());
    payloads.push(weights(&Shape::new([EMBED_DIM, HIDDEN_LEN])));
    payloads.push(weights(&Shape::new([HIDDEN_LEN, EMBED_DIM])));
    payloads.push(ones.clone());
    payloads.push(ones);
    payloads.push(weights(&Shape::new([EMBED_DIM, VOCABULARY_LEN])));
    payloads.push(weights(&Shape::new([VOCABULARY_LEN])));
    payloads
}

/// Returns the additive block-causal mask for the packed batch.
fn block_causal_mask(samples: usize) -> Tensor<f32> {
    let rows = samples * CONTEXT_LEN;
    let mut elements = Vec::with_capacity(rows * rows);
    for query in 0..rows {
        for key in 0..rows {
            let same_sample = query / CONTEXT_LEN == key / CONTEXT_LEN;
            let causal = key % CONTEXT_LEN <= query % CONTEXT_LEN;
            elements.push(if same_sample && causal {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Tensor::new([rows, rows], elements)
}

/// One emitted formulation of the training step: the module text,
/// the tape statistics, and the oracle's first-step loss.
struct Formulation {
    module: String,
    forward_nodes: usize,
    total_nodes: usize,
    oracle_loss: f32,
}

/// Records the transformer training step — head loop or batched
/// attention per `batched` — differentiates it, pins the result
/// order, compiles, and emits. Parameters are handed in so both
/// formulations share their payloads exactly.
fn recorded(
    payloads: &[Tensor<f32>],
    feeds: &(Tensor<f32>, Tensor<f32>),
    batched: bool,
) -> Formulation {
    let tape = Tape::new();
    let mut supply = payloads.iter().cloned();
    let mut next = move || supply.next().expect("payloads cover the model");

    let embeddings = tape.parameter(next());
    let positions_table = tape.parameter(next());
    let heads: Vec<[Value<'_, Tensor<f32>>; 3]> = (0..HEAD_COUNT)
        .map(|_| {
            [
                tape.parameter(next()),
                tape.parameter(next()),
                tape.parameter(next()),
            ]
        })
        .collect();
    let projection = tape.parameter(next());
    let attention_norm = RmsNorm::new(&tape, next(), Tensor::filled([], 1e-5));
    let hidden_weights = tape.parameter(next());
    let output_weights = tape.parameter(next());
    let hidden_norm = RmsNorm::new(&tape, next(), Tensor::filled([], 1e-5));
    let final_norm = RmsNorm::new(&tape, next(), Tensor::filled([], 1e-5));
    let logit_weights = tape.parameter(next());
    let logit_bias = tape.parameter(next());
    let scale = tape.leaf(Tensor::filled([], 1.0 / (HEAD_DIM as f32).sqrt()));

    let tokens = tape.input(Tensor::selection(vec![0; PACKED_LEN], VOCABULARY_LEN, 1.0));
    let positions = tape.leaf(Tensor::selection(
        (0..PACKED_LEN)
            .map(|row| row % CONTEXT_LEN)
            .collect::<Vec<_>>(),
        CONTEXT_LEN,
        1.0,
    ));
    let mask = tape.leaf(block_causal_mask(BATCH_LEN));
    let extraction = tape.leaf(Tensor::selection(
        (0..BATCH_LEN)
            .map(|sample| sample * CONTEXT_LEN + CONTEXT_LEN - 1)
            .collect::<Vec<_>>(),
        PACKED_LEN,
        1.0,
    ));
    let targets = tape.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));

    let stream = embeddings.gather(tokens) + positions_table.gather(positions);
    let normalized = attention_norm.express(&tape, stream);
    let attended = if batched {
        let project = |slot: usize| {
            let slices: Vec<_> = heads
                .iter()
                .map(|head| normalized.matmul(head[slot]))
                .collect();
            stack(&slices, 0)
        };
        let scores = project(0).matmul(project(1).permute([0, 2, 1]));
        let scaled = scores * scale.broadcast_like(scores);
        let weights = (scaled + mask.broadcast_along(0, scaled)).softmax(2);
        let context = weights.matmul(project(2));
        context.permute([1, 0, 2]).reshape([PACKED_LEN, EMBED_DIM])
    } else {
        let outputs: Vec<_> = heads
            .iter()
            .map(|head| {
                let scores = normalized
                    .matmul(head[0])
                    .matmul(normalized.matmul(head[1]).transpose());
                let scaled = scores * scale.broadcast_like(scores);
                let weights = (scaled + mask).softmax(1);
                weights.matmul(normalized.matmul(head[2]))
            })
            .collect();
        concat(&outputs, 1)
    };
    let stream = stream + attended.matmul(projection);
    let normalized = hidden_norm.express(&tape, stream);
    let stream = stream
        + normalized
            .matmul(hidden_weights)
            .relu()
            .matmul(output_weights);
    let states = final_norm.express(&tape, stream);
    let product = states.gather(extraction).matmul(logit_weights);
    let loss = cross_entropy(product + logit_bias.broadcast_along(0, product), targets);

    let forward_nodes = tape.len();
    let wrt: Vec<Symbol> = std::iter::once(embeddings)
        .chain([positions_table])
        .chain(heads.iter().flatten().copied())
        .chain([
            projection,
            hidden_weights,
            output_weights,
            logit_weights,
            logit_bias,
        ])
        .map(|value| value.symbol())
        .collect();
    let gradients = tape.differentiate(loss, wrt);
    let aliases: Vec<Symbol> = gradients
        .iter()
        .map(|&gradient| {
            let value = tape.resolve(gradient);
            value.reshape(value.shape()).symbol()
        })
        .collect();
    let (tokens, targets, loss) = (tokens.symbol(), targets.symbol(), loss.symbol());
    let network = tape.into_network();
    let parameters = network.parameters();
    let plan = network.compile(Request::roots(
        std::iter::once(loss).chain(aliases.iter().copied()),
    ));
    let run = plan.forward(
        &parameters,
        [(tokens, feeds.0.clone()), (targets, feeds.1.clone())],
    );
    Formulation {
        module: plan.emit_stablehlo().expect("the joint step emits"),
        forward_nodes,
        total_nodes: network.len(),
        oracle_loss: run.of(loss).to_vec()[0],
    }
}

/// Serves one module and times repeated identical requests.
fn timed(module: &str, name: &str, request: &[u8], response_len: usize) -> (f32, f64) {
    let directory = std::env::temp_dir().join("topos-attention-grading");
    std::fs::create_dir_all(&directory).expect("the staging directory creates");
    let module_path = directory.join(format!("{name}.mlir"));
    let static_path = directory.join("static.bin");
    let manifest_path = directory.join("manifest.json");
    std::fs::write(&module_path, module).expect("the module writes");
    std::fs::write(&static_path, []).expect("the empty static file writes");
    let mut shapes: Vec<String> = parameter_payloads()
        .iter()
        .map(|payload| {
            let extents: Vec<String> = payload
                .shape()
                .axes()
                .iter()
                .map(usize::to_string)
                .collect();
            format!("[{}]", extents.join(", "))
        })
        .collect();
    shapes.push(format!("[{PACKED_LEN}, {VOCABULARY_LEN}]"));
    shapes.push(format!("[{BATCH_LEN}, {VOCABULARY_LEN}]"));
    std::fs::write(
        &manifest_path,
        format!("{{\"dynamic\": [{}]}}", shapes.join(", ")),
    )
    .expect("the manifest writes");

    let python = std::env::var("TOPOS_XLA_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let mut command: Vec<String> = python.split_whitespace().map(str::to_string).collect();
    command.push("tools/serve-stablehlo-xla.py".to_string());
    let mut child: Child = Command::new(&command[0])
        .args(&command[1..])
        .arg(&module_path)
        .arg(&static_path)
        .arg(&manifest_path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("the serving process starts; is `jax` installed for it?");
    let mut requests: ChildStdin = child.stdin.take().expect("the server's input pipes");
    let mut responses: ChildStdout = child.stdout.take().expect("the server's output pipes");

    let mut step = |request: &[u8]| -> Vec<f32> {
        requests.write_all(request).expect("the request writes");
        requests.flush().expect("the request flushes");
        let mut response = vec![0u8; 4 * response_len];
        responses
            .read_exact(&mut response)
            .expect("the server answers; see its standard error");
        response
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect()
    };

    let first = step(request)[0];
    for _ in 0..WARMUP_STEPS {
        step(request);
    }
    let clock = Instant::now();
    for _ in 0..TIMED_STEPS {
        step(request);
    }
    let seconds = clock.elapsed().as_secs_f64();
    let _ = child.kill();
    let _ = child.wait();
    (first, seconds)
}

fn main() {
    let names = load_names();
    let mut samples = training_samples::<CONTEXT_LEN>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);

    // One fixed batch: identical feeds for the oracle and every
    // server request, so the two modules answer the same question.
    let batch = &samples[..BATCH_LEN];
    let tokens: Vec<usize> = batch
        .iter()
        .flat_map(|(context, _)| context.iter().copied())
        .collect();
    let targets: Vec<usize> = batch.iter().map(|&(_, next)| next).collect();
    let feeds = (
        Tensor::selection(tokens, VOCABULARY_LEN, 1.0_f32),
        Tensor::selection(targets, VOCABULARY_LEN, 1.0_f32),
    );

    let payloads = parameter_payloads();
    let request: Vec<u8> = payloads
        .iter()
        .flat_map(|payload| payload.to_vec())
        .chain(feeds.0.to_vec())
        .chain(feeds.1.to_vec())
        .flat_map(f32::to_le_bytes)
        .collect();

    for (name, batched) in [("head-loop", false), ("batched", true)] {
        let formulation = recorded(&payloads, &feeds, batched);
        let response_len = 1 + payloads
            .iter()
            .enumerate()
            .filter(|&(index, _)| !is_norm_gain(index))
            .map(|(_, payload)| payload.shape().volume())
            .sum::<usize>();
        let dots = formulation.module.matches("dot_general").count();
        let (first, seconds) = timed(&formulation.module, name, &request, response_len);
        let drift = (f64::from(formulation.oracle_loss) - f64::from(first)).abs()
            / f64::from(formulation.oracle_loss);
        // A red row on a flaky target is a finding, not a crash:
        // `TOPOS_ENVELOPE=report` keeps timing while naming the
        // deviation, the manifest posture for jax-metal.
        if std::env::var("TOPOS_ENVELOPE").as_deref() == Ok("report") {
            if drift >= 1e-3 {
                println!("{name}: RED ROW — left the oracle envelope, drift {drift:.1e}");
            }
        } else {
            assert!(drift < 1e-3, "{name} left the oracle envelope: {drift:e}");
        }
        println!(
            "{name:9} tape {} -> {} nodes, {} module lines, {dots} dot_general; \
             first loss {first:.6} (oracle drift {drift:.1e}); \
             {:.3} ms/step over {TIMED_STEPS} steps",
            formulation.forward_nodes,
            formulation.total_nodes,
            formulation.module.lines().count(),
            seconds * 1000.0 / TIMED_STEPS as f64,
        );
    }
}

/// Returns whether the payload at `index` is a norm gain, which stays
/// out of the `wrt` set and therefore out of the response.
fn is_norm_gain(index: usize) -> bool {
    let after_heads = 2 + 3 * HEAD_COUNT;
    index == after_heads + 1 || index == after_heads + 4 || index == after_heads + 5
}
