//! Trains the same character-level MLP as `makemore_mlp`, with the
//! hand-rolled layers replaced by the [`Mlp`] facade: the embedding
//! stage stays explicit (`gather` plus `reshape`, which the facade does
//! not cover), and the facade records the tanh hidden layer and the
//! affine output.
//!
//! The seeds, the parameter allocation order, and the batches all match
//! `makemore_mlp`, and the facade records the same operations the
//! hand-rolled model does, so the two examples train identically: the
//! facade is packaging, not different math. Expressing the facade twice
//! — once batch-shaped for training, once single-row for sampling —
//! replaces the hand-rolled twin expression.
//!
//! Run with: `cargo run --release --example makemore_mlp_facade`

mod corpus;

use poorgrad::{Mlp, Network, Shape, Tensor, Tensorial, cross_entropy, init};

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

fn main() {
    let names = load_names();
    let mut samples = training_samples::<CONTEXT_LEN>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);
    println!("loaded {} names, {} samples", names.len(), samples.len());

    let network: Network<Tensor<f32>> = Network::new();

    // The embedding table stays a plain parameter; the facade covers
    // the dense layers only. The allocation order and seeds match
    // `makemore_mlp`, so both examples start from identical weights.
    let embeddings = network.parameter(init::normal(8, 1.0)(&Shape::new([
        VOCABULARY_LEN,
        EMBED_DIM,
    ])));
    let mlp = Mlp::new(
        &network,
        &[CONTEXT_LEN * EMBED_DIM, HIDDEN_LEN, VOCABULARY_LEN],
        init::xavier(7),
    );

    // The training expression, batch-shaped: contexts and targets are
    // one-hot selections fed per run, the defaults only fix the shapes.
    let contexts = network.input(Tensor::selection(
        vec![0; BATCH_LEN * CONTEXT_LEN],
        VOCABULARY_LEN,
        1.0,
    ));
    let targets = network.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));
    let embedded = embeddings
        .gather(contexts)
        .reshape([BATCH_LEN, CONTEXT_LEN * EMBED_DIM]);
    let loss = cross_entropy(mlp.express(&network, embedded), targets);

    // The sampling twin: the same parameters expressed over a single
    // context row, with the composite softmax on top.
    let sample_context =
        network.input(Tensor::selection(vec![0; CONTEXT_LEN], VOCABULARY_LEN, 1.0));
    let sample_embedded = embeddings
        .gather(sample_context)
        .reshape([1, CONTEXT_LEN * EMBED_DIM]);
    let sample_probabilities = mlp.express(&network, sample_embedded).softmax(1);

    let contexts_symbol = contexts.symbol();
    let targets_symbol = targets.symbol();
    let loss_symbol = loss.symbol();
    let sample_context_symbol = sample_context.symbol();
    let sample_probabilities_symbol = sample_probabilities.symbol();
    let recorded_nodes = network.len();

    // A fresh model is roughly uniform over the vocabulary, so the
    // first printed loss should sit near `ln(27) ~ 3.30`; the goal is
    // to push below the bigram limit of ~2.45.
    let fast = Tensor::new([], [0.1]);
    let slow = Tensor::new([], [0.01]);
    let mut network = network;
    let mut window_loss = 0.0;
    for step in 0..5000 {
        let start = (step * BATCH_LEN) % (samples.len() - BATCH_LEN);
        let batch = &samples[start..start + BATCH_LEN];
        let batch_contexts: Vec<usize> = batch
            .iter()
            .flat_map(|(context, _)| context.iter().copied())
            .collect();
        let batch_targets: Vec<usize> = batch.iter().map(|&(_, next)| next).collect();

        let loss_value = network.resolve(loss_symbol);
        let evaluation = network.forward_with([
            (
                contexts_symbol,
                Tensor::selection(batch_contexts, VOCABULARY_LEN, 1.0),
            ),
            (
                targets_symbol,
                Tensor::selection(batch_targets, VOCABULARY_LEN, 1.0),
            ),
        ]);
        let batch_loss = evaluation.of(loss_value).to_vec()[0];
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
        let gradients = evaluation.backward(loss_value);
        let learning_rate = if step < 4000 { &fast } else { &slow };
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    assert_eq!(network.len(), recorded_nodes);
    println!("the tape held {recorded_nodes} nodes through every step");

    println!("sampled names:");
    let mut state: u64 = 7;
    for _ in 0..10 {
        let mut window = [0usize; CONTEXT_LEN];
        let mut name = String::new();
        loop {
            let evaluation = network.forward_with([(
                sample_context_symbol,
                Tensor::selection(window.to_vec(), VOCABULARY_LEN, 1.0),
            )]);
            let row = evaluation
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
