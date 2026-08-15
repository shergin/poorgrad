//! Trains the `makemore_mlp` model with its backward pass recorded —
//! the chain rule as tape instead of engine procedure. The model, the
//! seeds, and the batch schedule are identical to `makemore_mlp`, so
//! the two examples train bit-identically; what changes is where the
//! gradients come from: `Network::differentiate` appends them to the
//! tape once, one forward-only plan compiles `[loss, gradients...]`
//! together, and every training step is a single plan run — no
//! backward pass executes at all.
//!
//! The recorded gradients are ordinary values, so everything the plan
//! tier offers applies to them unchanged: dead-node elimination drops
//! the sampling twin, buffer liveness frees intermediates the gradient
//! no longer needs, and `Plan::describe()` prints the whole chain rule
//! as a schedule.
//!
//! Run with: `cargo run --release --example makemore_mlp_compiled`

mod chart;
mod corpus;

use std::time::Instant;

use poorgrad::{Network, Shape, Tensor, Tensorial, Value, cross_entropy, init};

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

/// The model's parameters as recorded proxies, laid out exactly as in
/// `makemore_mlp` so the two examples share their seeds.
struct Model<'network> {
    embeddings: Value<'network, Tensor<f32>>,
    hidden_weights: Value<'network, Tensor<f32>>,
    hidden_bias: Value<'network, Tensor<f32>>,
    output_weights: Value<'network, Tensor<f32>>,
    output_bias: Value<'network, Tensor<f32>>,
}

impl<'network> Model<'network> {
    /// Allocates the parameters on `network`: an embedding table, one
    /// tanh hidden layer, and an affine output layer, Xavier-scaled
    /// with zero biases.
    fn new(network: &'network Network<Tensor<f32>>) -> Self {
        let mut weights = init::xavier(7);
        Self {
            embeddings: network.parameter(init::normal(8, 1.0)(&Shape::new([
                VOCABULARY_LEN,
                EMBED_DIM,
            ]))),
            hidden_weights: network
                .parameter(weights(&Shape::new([CONTEXT_LEN * EMBED_DIM, HIDDEN_LEN]))),
            hidden_bias: network.parameter(weights(&Shape::new([HIDDEN_LEN]))),
            output_weights: network.parameter(weights(&Shape::new([HIDDEN_LEN, VOCABULARY_LEN]))),
            output_bias: network.parameter(weights(&Shape::new([VOCABULARY_LEN]))),
        }
    }

    /// Returns the parameters in a fixed order, for pairing with their
    /// recorded gradients.
    fn parameters(&self) -> [Value<'network, Tensor<f32>>; 5] {
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
        contexts: Value<'network, Tensor<f32>>,
        rows: usize,
    ) -> Value<'network, Tensor<f32>> {
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

    let network = Network::new();
    let model = Model::new(&network);

    // The training expression, batch-shaped: contexts and targets are
    // one-hot selections fed per run, the defaults only fix the shapes.
    let contexts = network.input(Tensor::selection(
        vec![0; BATCH_LEN * CONTEXT_LEN],
        VOCABULARY_LEN,
        1.0,
    ));
    let targets = network.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));
    let loss = cross_entropy(model.express(contexts, BATCH_LEN), targets);

    // The sampling twin: the same parameters expressed over a single
    // context row, with the composite softmax on top.
    let sample_context =
        network.input(Tensor::selection(vec![0; CONTEXT_LEN], VOCABULARY_LEN, 1.0));
    let sample_probabilities = model.express(sample_context, 1).softmax(1);

    let contexts_symbol = contexts.symbol();
    let targets_symbol = targets.symbol();
    let loss_symbol = loss.symbol();
    let sample_context_symbol = sample_context.symbol();
    let sample_probabilities_symbol = sample_probabilities.symbol();

    // The whole difference from `makemore_mlp` is these lines: the
    // chain rule is appended to the tape once, and one forward-only
    // plan serves loss and gradients together for every generation.
    // The sampling twin is not among the targets, so the plan's
    // dead-node elimination drops it from every training step.
    let forward_nodes = network.len();
    let parameter_symbols = model.parameters().map(|parameter| parameter.symbol());
    let gradient_symbols = network.differentiate(loss_symbol, parameter_symbols);
    println!(
        "recorded the chain rule: {} forward nodes + {} gradient nodes",
        forward_nodes,
        network.len() - forward_nodes
    );
    let plan = network.compile(
        std::iter::once(loss_symbol).chain(gradient_symbols.iter().copied()),
        [],
    );

    // A fresh model is roughly uniform over the vocabulary, so the
    // first printed loss should sit near `ln(27) ~ 3.30`; matched
    // seeds make every printed number equal `makemore_mlp`'s.
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

        let run = plan.forward(
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
        let batch_loss = run.of(network.resolve(loss_symbol)).to_vec()[0];
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
        // The gradients were computed by the same plan run that
        // computed the loss; assembling the update direction is a
        // read, not a backward pass.
        let gradients =
            run.recorded_gradients(parameter_symbols.iter().zip(&gradient_symbols).map(
                |(&parameter, &gradient)| (network.resolve(parameter), network.resolve(gradient)),
            ));
        let learning_rate = if step < 4000 { &fast } else { &slow };
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    println!(
        "trained {} steps in {:.3}s, one plan run per step, no backward pass",
        losses.len(),
        training.elapsed().as_secs_f64()
    );
    println!("{}", loss_chart("compiled mlp training", &losses));

    println!("sampled names:");
    let mut state: u64 = 7;
    for _ in 0..10 {
        let mut window = [0usize; CONTEXT_LEN];
        let mut name = String::new();
        loop {
            let run = network.forward_for(
                [sample_probabilities_symbol],
                [(
                    sample_context_symbol,
                    Tensor::selection(window.to_vec(), VOCABULARY_LEN, 1.0),
                )],
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
