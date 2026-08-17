//! Trains the character-level MLP with batch normalization —
//! makemore's third act: the hidden preactivation is standardized by
//! its minibatch statistics before the tanh, so the squash stays in
//! its active range whatever the initialization does. The hidden
//! layer loses its bias (the norm's learned shift replaces it), and
//! the norm's scale and shift train like any other parameters.
//!
//! The two-tape idiom carries the norm's two modes: the training
//! expression normalizes by the batch's own statistics
//! ([`BatchNorm::express`]) and the single-row sampling twin
//! normalizes by running estimates fed per run
//! ([`BatchNorm::express_with`]). The running estimates live in the
//! training loop as plain payloads, maintained as an exponential
//! moving average of the batch statistics — which the loop reads
//! through the training plan's keep-set, the declared-observability
//! contract doing exactly what it was built for.
//!
//! Run with: `cargo run --release --example makemore_mlp_batchnorm`

mod chart;
mod corpus;

use std::time::Instant;

use topos::{BatchNorm, Compile, Network, Shape, Tensor, Tensorial, Value, cross_entropy, init};

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

/// The exponential-moving-average momentum of the running statistics:
/// each step keeps `1 - MOMENTUM` of the estimate and takes `MOMENTUM`
/// of the batch statistic.
const MOMENTUM: f32 = 0.01;

/// The model's parameters as recorded proxies: the part-2 layout with
/// the hidden bias replaced by a batch-normalization stage.
struct Model<'network> {
    embeddings: Value<'network, Tensor<f32>>,
    hidden_weights: Value<'network, Tensor<f32>>,
    norm: BatchNorm<Tensor<f32>>,
    output_weights: Value<'network, Tensor<f32>>,
    output_bias: Value<'network, Tensor<f32>>,
}

impl<'network> Model<'network> {
    /// Allocates the parameters on `network`: the embedding table, the
    /// bias-free hidden layer, the norm's scale and shift (ones and
    /// zeros, the standard start), and the affine output.
    fn new(network: &'network Network<Tensor<f32>>) -> Self {
        let mut weights = init::xavier(7);
        Self {
            embeddings: network.parameter(init::normal(8, 1.0)(&Shape::new([
                VOCABULARY_LEN,
                EMBED_DIM,
            ]))),
            hidden_weights: network
                .parameter(weights(&Shape::new([CONTEXT_LEN * EMBED_DIM, HIDDEN_LEN]))),
            norm: BatchNorm::new(
                network,
                Tensor::filled([HIDDEN_LEN], 1.0),
                Tensor::filled([HIDDEN_LEN], 0.0),
                Tensor::filled([], 1e-5),
            ),
            output_weights: network.parameter(weights(&Shape::new([HIDDEN_LEN, VOCABULARY_LEN]))),
            output_bias: network.parameter(weights(&Shape::new([VOCABULARY_LEN]))),
        }
    }

    /// Records the shared head of both expressions: embed, flatten,
    /// and the bias-free hidden preactivation.
    fn preactivation(
        &self,
        contexts: Value<'network, Tensor<f32>>,
        rows: usize,
    ) -> Value<'network, Tensor<f32>> {
        self.embeddings
            .gather(contexts)
            .reshape([rows, CONTEXT_LEN * EMBED_DIM])
            .matmul(self.hidden_weights)
    }

    /// Records the shared tail: squash the normalized preactivation
    /// and score.
    fn logits(&self, normalized: Value<'network, Tensor<f32>>) -> Value<'network, Tensor<f32>> {
        let hidden = normalized.tanh();
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

    let network = Network::new();
    let model = Model::new(&network);

    // The training expression: the norm standardizes each feature by
    // the minibatch's own statistics, and hands them back for the
    // running estimates.
    let contexts = network.input(Tensor::selection(
        vec![0; BATCH_LEN * CONTEXT_LEN],
        VOCABULARY_LEN,
        1.0,
    ));
    let targets = network.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));
    let normalization = model
        .norm
        .express(&network, model.preactivation(contexts, BATCH_LEN));
    let loss = cross_entropy(model.logits(normalization.output), targets);

    // The sampling twin normalizes by running estimates fed per run:
    // the norm's inference mode, one recorded expression for every
    // generation of the estimates.
    let sample_context =
        network.input(Tensor::selection(vec![0; CONTEXT_LEN], VOCABULARY_LEN, 1.0));
    let running_mean = network.input(Tensor::filled([HIDDEN_LEN], 0.0));
    let running_variance = network.input(Tensor::filled([HIDDEN_LEN], 1.0));
    let sample_normalized = model.norm.express_with(
        &network,
        model.preactivation(sample_context, 1),
        running_mean,
        running_variance,
    );
    let sample_probabilities = model.logits(sample_normalized).softmax(1);

    let contexts_symbol = contexts.symbol();
    let targets_symbol = targets.symbol();
    let loss_symbol = loss.symbol();
    let mean_symbol = normalization.mean.symbol();
    let variance_symbol = normalization.variance.symbol();
    let sample_context_symbol = sample_context.symbol();
    let running_mean_symbol = running_mean.symbol();
    let running_variance_symbol = running_variance.symbol();
    let sample_probabilities_symbol = sample_probabilities.symbol();
    let recorded_nodes = network.len();

    // Compile once: the training plan keeps the batch statistics
    // readable — the keep-set naming exactly what the loop reads —
    // and the sampling plan is forward-only.
    let training_plan = network.compile(
        Compile::roots([loss_symbol])
            .observe([mean_symbol, variance_symbol])
            .engine_backward(),
    );
    let sampling_plan = network.compile(Compile::roots([sample_probabilities_symbol]));

    // The running estimates: loop-owned payloads, never engine state.
    let mut mean_estimate = Tensor::filled([HIDDEN_LEN], 0.0_f32);
    let mut variance_estimate = Tensor::filled([HIDDEN_LEN], 1.0_f32);
    let keep = Tensor::filled([HIDDEN_LEN], 1.0 - MOMENTUM);
    let take = Tensor::filled([HIDDEN_LEN], MOMENTUM);

    let fast = Tensor::new([], [0.1]);
    let slow = Tensor::new([], [0.01]);
    let mut network = network;
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

        let loss_value = network.resolve(loss_symbol);
        let run = training_plan.forward(
            &network,
            [
                (
                    contexts_symbol,
                    Tensor::selection(batch_contexts, VOCABULARY_LEN, 1.0),
                ),
                (
                    targets_symbol,
                    Tensor::selection(batch_targets, VOCABULARY_LEN, 1.0),
                ),
            ],
        );
        let batch_loss = run.of(loss_value).to_vec()[0];
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

        // The running estimates fold in this batch's statistics, read
        // through the keep-set: payload arithmetic in loop land.
        let batch_mean = run.of(network.resolve(mean_symbol)).clone();
        let batch_variance = run.of(network.resolve(variance_symbol)).clone();
        mean_estimate = mean_estimate * keep.clone() + batch_mean * take.clone();
        variance_estimate = variance_estimate * keep.clone() + batch_variance * take.clone();

        let gradients = run.backward(loss_value);
        let learning_rate = if step < 4000 { &fast } else { &slow };
        network = network.update(&gradients, |parameter, gradient| {
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
    println!("{}", loss_chart("mlp + batchnorm training", &losses));

    println!("sampled names (running statistics fed per draw):");
    let mut state: u64 = 7;
    for _ in 0..10 {
        let mut window = [0usize; CONTEXT_LEN];
        let mut name = String::new();
        loop {
            let run = sampling_plan.forward(
                &network,
                [
                    (
                        sample_context_symbol,
                        Tensor::selection(window.to_vec(), VOCABULARY_LEN, 1.0),
                    ),
                    (running_mean_symbol, mean_estimate.clone()),
                    (running_variance_symbol, variance_estimate.clone()),
                ],
            );
            let row = run
                .of(network.resolve(sample_probabilities_symbol))
                .to_vec();
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
