//! Trains the `makemore_mlp` model twice — once with plain SGD, once
//! with Adam — and draws both loss curves on one chart: the optimizer
//! act. The model, seeds, and batch schedule are identical to
//! `makemore_mlp_compiled`, gradients come from the same recorded
//! route (`differentiate` + one forward-only plan), and the SGD run
//! reproduces that example's losses bit for bit; the only new moving
//! part is which instrument the loop hands the gradients to.
//!
//! Optimizers are caller-owned instruments: the loop owns the
//! learning-rate arithmetic and the parameter state it steps, and any
//! `Optimizer` implementation slots into the same line — the
//! comparison below iterates strategies through `&mut dyn Optimizer`,
//! the sanctioned example-side use of dynamic dispatch.
//!
//! `TOPOS_GRADIENTS=engine` flips the gradient source to the
//! interpreter's `backward`. The losses are bit-identical either way
//! (the parity contract), but the gradient fields are not the same
//! shape of object: recorded gradients reach the optimizer with
//! O(1)-constant payloads in every non-parameter slot, so Adam's
//! moment fields stay constant there too, while engine backward
//! carries dense intermediate gradients into the moments. Run both
//! under a memory monitor to see the difference.
//!
//! Run with: `cargo run --release --example makemore_mlp_adam`

// The shared corpus module also serves the sampling examples; this act
// never samples, so its `draw`/`from_token` stay unused here.
#[allow(dead_code)]
mod corpus;

use std::time::Instant;

use malevich::stat::Window;
use malevich::{Frame, Line, Plot, Rule};
use topos::{
    Adam, Compile, Optimizer, Sgd, Shape, Symbol, Tape, Tensor, Value, cross_entropy, init,
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

/// How many training steps each optimizer runs.
const STEPS: usize = 5000;

/// The corpus's bigram limit, the line both curves aim below.
const BIGRAM_LIMIT: f64 = 2.45;

/// The model's parameters as recorded proxies, laid out exactly as in
/// `makemore_mlp_compiled` so the runs share their seeds.
struct Model<'tape> {
    embeddings: Value<'tape, Tensor<f32>>,
    hidden_weights: Value<'tape, Tensor<f32>>,
    hidden_bias: Value<'tape, Tensor<f32>>,
    output_weights: Value<'tape, Tensor<f32>>,
    output_bias: Value<'tape, Tensor<f32>>,
}

impl<'tape> Model<'tape> {
    /// Allocates the parameters on `tape`: an embedding table, one
    /// tanh hidden layer, and an affine output layer, Xavier-scaled
    /// with zero biases.
    fn new(tape: &'tape Tape<Tensor<f32>>) -> Self {
        let mut weights = init::xavier(7);
        Self {
            embeddings: tape.parameter(init::normal(8, 1.0)(&Shape::new([
                VOCABULARY_LEN,
                EMBED_DIM,
            ]))),
            hidden_weights: tape
                .parameter(weights(&Shape::new([CONTEXT_LEN * EMBED_DIM, HIDDEN_LEN]))),
            hidden_bias: tape.parameter(weights(&Shape::new([HIDDEN_LEN]))),
            output_weights: tape.parameter(weights(&Shape::new([HIDDEN_LEN, VOCABULARY_LEN]))),
            output_bias: tape.parameter(weights(&Shape::new([VOCABULARY_LEN]))),
        }
    }

    /// Returns the parameters in a fixed order, for pairing with their
    /// recorded gradients.
    fn parameters(&self) -> [Value<'tape, Tensor<f32>>; 5] {
        [
            self.embeddings,
            self.hidden_weights,
            self.hidden_bias,
            self.output_weights,
            self.output_bias,
        ]
    }

    /// Records the model's expression over `contexts` — a one-hot
    /// `[rows * CONTEXT_LEN, vocab]` selection — and returns the
    /// `[rows, vocab]` logits: embed, flatten the context window,
    /// squash, and score.
    fn express(
        &self,
        contexts: Value<'tape, Tensor<f32>>,
        rows: usize,
    ) -> Value<'tape, Tensor<f32>> {
        let embedded = self
            .embeddings
            .gather(contexts)
            .reshape([rows, CONTEXT_LEN * EMBED_DIM]);
        let product = embedded.matmul(self.hidden_weights);
        let hidden = (product + self.hidden_bias.broadcast_along(0, product)).tanh();
        let product = hidden.matmul(self.output_weights);
        product + self.output_bias.broadcast_along(0, product)
    }
}

/// Trains a fresh model for `STEPS` under `optimizer`, with the
/// per-step learning rate from `learning_rate`, and returns the
/// per-step losses and the wall-clock seconds. Every run rebuilds the
/// same seeds, so trajectories differ only by the optimizer.
fn train(
    samples: &[([usize; CONTEXT_LEN], usize)],
    recorded: bool,
    optimizer: &mut dyn Optimizer<Tensor<f32>>,
    learning_rate: impl Fn(usize) -> Tensor<f32>,
) -> (Vec<f32>, f64) {
    let tape = Tape::new();
    let model = Model::new(&tape);
    let contexts = tape.input(Tensor::selection(
        vec![0; BATCH_LEN * CONTEXT_LEN],
        VOCABULARY_LEN,
        1.0,
    ));
    let targets = tape.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));
    let loss = cross_entropy(model.express(contexts, BATCH_LEN), targets);

    let (contexts, targets, loss) = (contexts.symbol(), targets.symbol(), loss.symbol());
    let parameter_symbols = model.parameters().map(|parameter| parameter.symbol());

    // The recorded route: the chain rule on the tape, one forward-only
    // plan per run. The engine route skips both and differentiates
    // each interpreter run procedurally — bit-identical losses, but
    // dense gradient fields where the recorded route carries
    // O(1)-constant non-parameter slots into the optimizer's moments.
    let gradient_symbols: Vec<Symbol> = if recorded {
        tape.differentiate(loss, parameter_symbols)
    } else {
        Vec::new()
    };
    let network = tape.into_network();
    let plan = recorded.then(|| {
        network.compile(Compile::roots(
            std::iter::once(loss).chain(gradient_symbols.iter().copied()),
        ))
    });

    let mut parameters = network.parameters();
    let mut losses = Vec::new();
    let training = Instant::now();
    for step in 0..STEPS {
        let start = (step * BATCH_LEN) % (samples.len() - BATCH_LEN);
        let batch = &samples[start..start + BATCH_LEN];
        let batch_contexts: Vec<usize> = batch
            .iter()
            .flat_map(|(context, _)| context.iter().copied())
            .collect();
        let batch_targets: Vec<usize> = batch.iter().map(|&(_, next)| next).collect();
        let feeds = [
            (
                contexts,
                Tensor::selection(batch_contexts, VOCABULARY_LEN, 1.0),
            ),
            (
                targets,
                Tensor::selection(batch_targets, VOCABULARY_LEN, 1.0),
            ),
        ];

        let (batch_loss, gradients) = if let Some(plan) = &plan {
            let run = plan.forward(&parameters, feeds);
            let gradients = run.recorded_gradients(
                parameter_symbols
                    .iter()
                    .copied()
                    .zip(gradient_symbols.iter().copied()),
            );
            (run.of(loss).to_vec()[0], gradients)
        } else {
            let run = network.forward(&parameters, feeds);
            (run.of(loss).to_vec()[0], run.backward(loss))
        };
        losses.push(batch_loss);

        parameters = optimizer.step(&parameters, &gradients, &learning_rate(step));
    }
    (losses, training.elapsed().as_secs_f64())
}

/// Renders both rolling-mean curves on one chart, with the bigram
/// limit as the line to beat.
fn comparison_chart(sgd: &[f32], adam: &[f32]) -> String {
    let window_len = (STEPS / 20).max(2);
    let smooth = |losses: &[f32]| {
        let losses: Vec<f64> = losses.iter().copied().map(f64::from).collect();
        Window::new(window_len).mean(&losses)
    };
    let sgd = smooth(sgd);
    let adam = smooth(adam);
    Plot::new()
        .layer(Line::y(&sgd[..]).label("sgd"))
        .layer(Line::y(&adam[..]).label("adam"))
        .layer(Rule::h(BIGRAM_LIMIT).label("bigram limit"))
        .title("sgd vs adam, rolling mean")
        .x_label("step")
        .y_label("loss")
        .render_best(&Frame::detect())
}

fn main() {
    let names = load_names();
    let mut samples = training_samples::<CONTEXT_LEN>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);
    println!("loaded {} names, {} samples", names.len(), samples.len());

    let recorded = match std::env::var("TOPOS_GRADIENTS").as_deref() {
        Ok("engine") => false,
        Ok("recorded") | Err(_) => true,
        Ok(other) => panic!("unknown TOPOS_GRADIENTS {other:?}; use recorded or engine"),
    };
    println!(
        "gradients: {}",
        if recorded {
            "recorded (differentiate + one forward-only plan)"
        } else {
            "engine (interpreter backward)"
        }
    );

    // SGD keeps `makemore_mlp_compiled`'s schedule, so its losses
    // reproduce that example bit for bit; Adam runs at one flat rate —
    // the adaptive moments replace the hand-tuned decay.
    let fast = Tensor::new([], [0.1_f32]);
    let slow = Tensor::new([], [0.01_f32]);
    let (sgd_losses, sgd_seconds) = train(&samples, recorded, &mut Sgd, |step| {
        if step < 4000 {
            fast.clone()
        } else {
            slow.clone()
        }
    });

    let mut adam = Adam::new(
        Tensor::new([], [0.9_f32]),
        Tensor::new([], [0.999_f32]),
        Tensor::new([], [1e-8_f32]),
    );
    let flat = Tensor::new([], [0.005_f32]);
    let (adam_losses, adam_seconds) = train(&samples, recorded, &mut adam, |_| flat.clone());

    for (label, losses, seconds) in [
        ("sgd", &sgd_losses, sgd_seconds),
        ("adam", &adam_losses, adam_seconds),
    ] {
        let window: f32 = losses[STEPS - 500..].iter().sum::<f32>() / 500.0;
        println!(
            "{label:5} {STEPS} steps in {seconds:.3}s ({:.2} ms/step), last-500 mean loss {window:.4}",
            seconds * 1000.0 / STEPS as f64
        );
    }
    println!("{}", comparison_chart(&sgd_losses, &adam_losses));
}
