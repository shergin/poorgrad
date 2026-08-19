//! Trains the `makemore_mlp` model through an emitted joint step —
//! E2, the compiled-training shape abroad: `differentiate` records
//! the chain rule, one forward-only plan compiles
//! `[loss, gradients...]`, `emit_stablehlo` turns it into one
//! StableHLO function `(parameters, batch) -> (loss, gradients...)`,
//! and XLA executes every training step while the host keeps the
//! update loop as plain payload arithmetic. The same binary first
//! trains the in-crate recorded plan — the oracle trajectory,
//! bit-identical to `makemore_mlp_compiled` — so the emitted run's
//! losses are graded against it on the page.
//!
//! Two E2 conventions from the design (`notes/emission.md` sec. 11):
//! result order is pinned by recording one same-shape `reshape`
//! alias per gradient in the order the caller wants results, and
//! training stages every argument dynamic, because parameters change
//! every step — the host sends them with the batch and never
//! writes them back onto the tape.
//!
//! Serving needs a Python with `jax` (`TOPOS_XLA_PYTHON` names
//! the interpreter; default `python3`); the backend follows jax's
//! own selection, CPU by default.
//!
//! Run with: `cargo run --release --example makemore_mlp_emitted`

// The shared corpus module also serves the sampling examples; this
// act never samples, so its `draw`/`from_token` stay unused here.
#[allow(dead_code)]
mod corpus;

use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use topos::{
    Differentiable, Request, Shape, Symbol, Tape, Tensor, Tensorial, Value, cross_entropy, init,
};

use corpus::{VOCABULARY_LEN, load_names, shuffle, training_samples};

/// How many characters of history the model sees before predicting the
/// next one.
const CONTEXT_LEN: usize = 3;

/// How many dimensions the character embedding space has.
const EMBED_DIM: usize = 10;

/// How many neurons the tanh hidden layer has.
const HIDDEN_LEN: usize = 100;

/// How many samples each training step feeds.
const BATCH_LEN: usize = 64;

/// How many training steps each route runs.
const STEPS: usize = 5000;

/// The initial parameter payloads, minted identically to
/// `makemore_mlp_compiled`'s `Model::new` (same factories, same call
/// order, same seeds), so both routes and that example share their
/// trajectory. Order is the recording order and therefore the
/// emitted module's leading-argument order.
fn initial_parameters() -> [Tensor<f32>; 5] {
    let mut weights = init::xavier(7);
    [
        init::normal(8, 1.0)(&Shape::new([VOCABULARY_LEN, EMBED_DIM])),
        weights(&Shape::new([CONTEXT_LEN * EMBED_DIM, HIDDEN_LEN])),
        weights(&Shape::new([HIDDEN_LEN])),
        weights(&Shape::new([HIDDEN_LEN, VOCABULARY_LEN])),
        weights(&Shape::new([VOCABULARY_LEN])),
    ]
}

/// Records the model's expression over `contexts` and returns the
/// `[rows, vocab]` logits: embed, flatten the context window, squash,
/// and score. `parameters` arrive in `initial_parameters` order.
fn express<'tape>(
    parameters: &[Value<'tape, Tensor<f32>>; 5],
    contexts: Value<'tape, Tensor<f32>>,
    rows: usize,
) -> Value<'tape, Tensor<f32>> {
    let [
        embeddings,
        hidden_weights,
        hidden_bias,
        output_weights,
        output_bias,
    ] = *parameters;
    let embedded = embeddings
        .gather(contexts)
        .reshape([rows, CONTEXT_LEN * EMBED_DIM]);
    let product = embedded.matmul(hidden_weights);
    let hidden = (product + hidden_bias.broadcast_along(0, product)).tanh();
    let product = hidden.matmul(output_weights);
    product + output_bias.broadcast_along(0, product)
}

/// One-hot payloads for a batch slice, contexts then targets.
fn batch_payloads(batch: &[([usize; CONTEXT_LEN], usize)]) -> (Tensor<f32>, Tensor<f32>) {
    let contexts: Vec<usize> = batch
        .iter()
        .flat_map(|(context, _)| context.iter().copied())
        .collect();
    let targets: Vec<usize> = batch.iter().map(|&(_, next)| next).collect();
    (
        Tensor::selection(contexts, VOCABULARY_LEN, 1.0),
        Tensor::selection(targets, VOCABULARY_LEN, 1.0),
    )
}

/// The resident XLA server over `tools/serve-stablehlo-xla.py`:
/// compile once, then one request per training step. Every argument
/// is dynamic — parameters change every step.
struct XlaServer {
    child: Child,
    requests: ChildStdin,
    responses: ChildStdout,
    response_len: usize,
}

impl XlaServer {
    fn new(module: &str, dynamic_shapes: &[&[usize]], response_len: usize) -> Self {
        let directory = std::env::temp_dir().join("topos-makemore-e2");
        std::fs::create_dir_all(&directory).expect("the staging directory creates");
        let module_path = directory.join("joint-step.mlir");
        let static_path = directory.join("static.bin");
        let manifest_path = directory.join("manifest.json");
        std::fs::write(&module_path, module).expect("the module writes");
        std::fs::write(&static_path, []).expect("the empty static file writes");
        let shapes: Vec<String> = dynamic_shapes
            .iter()
            .map(|shape| {
                let extents: Vec<String> = shape.iter().map(usize::to_string).collect();
                format!("[{}]", extents.join(", "))
            })
            .collect();
        std::fs::write(
            &manifest_path,
            format!("{{\"dynamic\": [{}]}}", shapes.join(", ")),
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
            response_len,
        }
    }

    /// Executes one joint step: sends every argument's elements in
    /// order, reads back the results in the pinned order.
    fn step(&mut self, arguments: impl Iterator<Item = f32>) -> Vec<f32> {
        let request: Vec<u8> = arguments.flat_map(f32::to_le_bytes).collect();
        self.requests
            .write_all(&request)
            .expect("the request writes");
        self.requests.flush().expect("the request flushes");
        let mut response = vec![0u8; 4 * self.response_len];
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

fn main() {
    let names = load_names();
    let mut samples = training_samples::<CONTEXT_LEN>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);
    println!("loaded {} names, {} samples", names.len(), samples.len());

    // Record the expression and the chain rule once. The network's
    // only job here is to be the spec: after `emit_stablehlo`, the
    // emitted route never touches it again.
    let tape = Tape::new();
    let initial = initial_parameters();
    let parameters = initial.clone().map(|payload| tape.parameter(payload));
    let contexts = tape.input(Tensor::selection(
        vec![0; BATCH_LEN * CONTEXT_LEN],
        VOCABULARY_LEN,
        1.0,
    ));
    let targets = tape.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));
    let loss = cross_entropy(express(&parameters, contexts, BATCH_LEN), targets);

    let (contexts, targets, loss) = (contexts.symbol(), targets.symbol(), loss.symbol());
    let parameter_symbols = parameters.map(|parameter| parameter.symbol());
    let gradient_symbols = tape.differentiate(loss, parameter_symbols);
    // Pin the emitted result order: results are the readable set in
    // recording order, and the raw gradient nodes sit at scan-artifact
    // positions, so record one same-shape reshape alias per gradient,
    // in parameter order, and compile the aliases as the roots.
    let aliases: Vec<Symbol> = gradient_symbols
        .iter()
        .map(|&gradient| {
            let value = tape.resolve(gradient);
            value.reshape(value.shape()).symbol()
        })
        .collect();
    let network = tape.into_network();
    let plan = network.compile(Request::roots(
        std::iter::once(loss).chain(aliases.iter().copied()),
    ));
    let module = plan.emit_stablehlo().expect("the joint step emits");
    println!(
        "emitted the joint step: {} nodes, {} bytes of StableHLO",
        plan.len(),
        module.len()
    );

    let fast = Tensor::new([], [0.1_f32]);
    let slow = Tensor::new([], [0.01_f32]);
    let learning_rate = |step: usize| if step < 4000 { &fast } else { &slow };

    // Route one, the oracle: the in-crate recorded plan, bit-identical
    // to `makemore_mlp_compiled`.
    let mut oracle_parameters = network.parameters();
    let mut oracle_losses = Vec::new();
    let oracle_clock = Instant::now();
    for step in 0..STEPS {
        let start = (step * BATCH_LEN) % (samples.len() - BATCH_LEN);
        let (batch_contexts, batch_targets) = batch_payloads(&samples[start..start + BATCH_LEN]);
        let run = plan.forward(
            &oracle_parameters,
            [(contexts, batch_contexts), (targets, batch_targets)],
        );
        oracle_losses.push(run.of(loss).to_vec()[0]);
        // The aliases are the readable set; they carry the gradient
        // payloads bit for bit (a same-shape reshape is the identity).
        let gradients = run.recorded_gradients(
            parameter_symbols
                .iter()
                .copied()
                .zip(aliases.iter().copied()),
        );
        let rate = learning_rate(step);
        oracle_parameters = oracle_parameters.step(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * rate.broadcast_like(gradient)
        });
    }
    let oracle_seconds = oracle_clock.elapsed().as_secs_f64();

    // Route two, abroad: the same module through XLA, parameters held
    // host-side as plain payloads and updated with payload arithmetic.
    let shapes: Vec<Vec<usize>> = initial
        .iter()
        .map(|tensor| tensor.shape().axes().to_vec())
        .chain([
            vec![BATCH_LEN * CONTEXT_LEN, VOCABULARY_LEN],
            vec![BATCH_LEN, VOCABULARY_LEN],
        ])
        .collect();
    let dynamic_shapes: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
    let gradient_volumes: Vec<usize> = initial
        .iter()
        .map(|tensor| tensor.shape().volume())
        .collect();
    let response_len = 1 + gradient_volumes.iter().sum::<usize>();
    let mut server = XlaServer::new(&module, &dynamic_shapes, response_len);

    let mut hosted: Vec<Tensor<f32>> = initial.to_vec();
    let mut emitted_losses = Vec::new();
    let emitted_clock = Instant::now();
    for step in 0..STEPS {
        let start = (step * BATCH_LEN) % (samples.len() - BATCH_LEN);
        let (batch_contexts, batch_targets) = batch_payloads(&samples[start..start + BATCH_LEN]);
        let response = server.step(
            hosted
                .iter()
                .flat_map(|tensor| tensor.to_vec())
                .chain(batch_contexts.to_vec())
                .chain(batch_targets.to_vec()),
        );
        emitted_losses.push(response[0]);
        let mut at = 1;
        let rate = learning_rate(step);
        for (parameter, &volume) in hosted.iter_mut().zip(&gradient_volumes) {
            let gradient = Tensor::new(
                parameter.shape().clone(),
                response[at..at + volume].to_vec(),
            );
            at += volume;
            *parameter = parameter.clone() - gradient.clone() * rate.broadcast_like(&gradient);
        }
    }
    let emitted_seconds = emitted_clock.elapsed().as_secs_f64();

    // Tier 1, per step: the first loss is one forward on identical
    // payloads — XLA may reassociate, never diverge.
    let first_drift = (f64::from(oracle_losses[0]) - f64::from(emitted_losses[0])).abs()
        / f64::from(oracle_losses[0]);
    println!(
        "step 0: oracle {:.6}, emitted {:.6} (relative drift {first_drift:.2e})",
        oracle_losses[0], emitted_losses[0]
    );
    assert!(
        first_drift < 1e-3,
        "the emitted first step left the oracle envelope"
    );

    // Tier 2, trajectory: windowed means over the whole schedule;
    // per-step drift compounds legitimately, so the page shows both.
    println!("           oracle   emitted");
    for window in 0..STEPS / 500 {
        let range = window * 500..(window + 1) * 500;
        let mean = |losses: &[f32]| losses[range.clone()].iter().sum::<f32>() / 500.0;
        println!(
            "steps {:4}..{:4}: {:.4}   {:.4}",
            range.start,
            range.end,
            mean(&oracle_losses),
            mean(&emitted_losses)
        );
    }
    println!(
        "oracle  (in-crate plan): {STEPS} steps in {oracle_seconds:.3}s ({:.2} ms/step)",
        oracle_seconds * 1000.0 / STEPS as f64
    );
    println!(
        "emitted (XLA joint step): {STEPS} steps in {emitted_seconds:.3}s ({:.2} ms/step)",
        emitted_seconds * 1000.0 / STEPS as f64
    );
}
