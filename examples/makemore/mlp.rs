//! Trains a character-level MLP language model on names — makemore's
//! second act, after the bigram: a three-character context embedded
//! into a small vector space and pushed through one tanh hidden layer
//! (Bengio et al., 2003), hand-rolled from raw parameters and graph
//! operations rather than the `Mlp` facade, so every moving part is on
//! the page. The `makemore_mlp_facade` example is the same model built
//! on the facade, and trains identically.
//!
//! The tape carries two expressions of the same parameters: a
//! batch-shaped one for training and a single-row twin for sampling,
//! because input shapes are baked in at recording time. Minibatches
//! arrive as per-run feeds, so the tape never grows during training.
//!
//! Run with: `cargo run --release --example makemore_mlp`

mod chart;
mod corpus;

use std::time::Instant;

use topos::{Shape, Tape, Tensor, Tensorial, Value, cross_entropy, init};

use chart::loss_chart;
use corpus::{VOCABULARY_LEN, draw, from_token, load_names, shuffle, training_samples};

/// How many characters of history the model sees before predicting the
/// next one.
const CONTEXT_LEN: usize = 3;

/// How many dimensions the character embedding space has.
const EMBED_DIM: usize = 10;

/// How many neurons the tanh hidden layer has.
const HIDDEN_LEN: usize = 100;

/// How many samples each training step feeds.
const BATCH_LEN: usize = 64;

/// The model's parameters as recorded proxies: what the `Mlp` facade
/// would hold for us, laid out by hand.
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

fn main() {
    let names = load_names();
    let mut samples = training_samples::<CONTEXT_LEN>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);
    println!("loaded {} names, {} samples", names.len(), samples.len());

    let tape = Tape::new();
    let model = Model::new(&tape);

    // The training expression, batch-shaped: contexts and targets are
    // one-hot selections fed per run, the defaults only fix the shapes.
    let contexts = tape.input(Tensor::selection(
        vec![0; BATCH_LEN * CONTEXT_LEN],
        VOCABULARY_LEN,
        1.0,
    ));
    let targets = tape.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));
    let loss = cross_entropy(model.express(contexts, BATCH_LEN), targets);

    // The sampling twin: the same parameters expressed over a single
    // context row, with the composite softmax on top.
    let sample_context = tape.input(Tensor::selection(vec![0; CONTEXT_LEN], VOCABULARY_LEN, 1.0));
    let sample_probabilities = model.express(sample_context, 1).softmax(1);

    let (contexts, targets, loss, sample_context, sample_probabilities) = (
        contexts.symbol(),
        targets.symbol(),
        loss.symbol(),
        sample_context.symbol(),
        sample_probabilities.symbol(),
    );
    let recorded_nodes = tape.len();
    let network = tape.into_network();
    let mut parameters = network.parameters();

    // A fresh model is roughly uniform over the vocabulary, so the
    // first printed loss should sit near `ln(27) ~ 3.30`; the goal is
    // to push below the bigram limit of ~2.45.
    let fast = Tensor::new([], [0.1]);
    let slow = Tensor::new([], [0.01]);
    let mut window_loss = 0.0;
    let mut losses = Vec::new();
    let training = Instant::now();
    for step in 0..5000 {
        let start = (step * BATCH_LEN) % (samples.len() - BATCH_LEN);
        let batch = &samples[start..start + BATCH_LEN];
        let batch_contexts: Vec<usize> = batch
            .iter()
            .flat_map(|(context, _)| context.iter().copied())
            .collect();
        let batch_targets: Vec<usize> = batch.iter().map(|&(_, next)| next).collect();

        // Slice the run to the loss: the sampling twin on the same
        // tape is skipped during training.
        let run = network.forward_for(
            &parameters,
            [loss],
            [
                (
                    contexts,
                    Tensor::selection(batch_contexts, VOCABULARY_LEN, 1.0),
                ),
                (
                    targets,
                    Tensor::selection(batch_targets, VOCABULARY_LEN, 1.0),
                ),
            ],
        );
        let batch_loss = run.of(loss).to_vec()[0];
        losses.push(batch_loss);
        if step == 0 {
            println!(
                "step 0: minibatch loss = {batch_loss:.4} (a uniform model costs ln 27 ~ 3.30)"
            );
        }
        window_loss += batch_loss;
        if (step + 1) % 500 == 0 {
            println!(
                "steps {:4}..{:4}: mean minibatch loss = {:.4}",
                step + 1 - 500,
                step + 1,
                window_loss / 500.0
            );
            window_loss = 0.0;
        }
        let gradients = run.backward(loss);
        let learning_rate = if step < 4000 { &fast } else { &slow };
        parameters = parameters.step(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    println!(
        "trained {} steps in {:.3}s",
        losses.len(),
        training.elapsed().as_secs_f64()
    );

    assert_eq!(network.len(), recorded_nodes);
    println!("the tape held {recorded_nodes} nodes through every step");
    println!("{}", loss_chart("mlp training", &losses));

    println!("sampled names:");
    let mut state: u64 = 7;
    for _ in 0..10 {
        let mut window = [0usize; CONTEXT_LEN];
        let mut name = String::new();
        loop {
            let run = network.forward_for(
                &parameters,
                [sample_probabilities],
                [(
                    sample_context,
                    Tensor::selection(window.to_vec(), VOCABULARY_LEN, 1.0),
                )],
            );
            let row = run.of(sample_probabilities).to_vec();
            let token = draw(&row, &mut state);
            if token == 0 {
                break;
            }
            name.push(from_token(token));
            window.rotate_left(1);
            window[CONTEXT_LEN - 1] = token;
        }
        println!("  {name}");
    }
}
