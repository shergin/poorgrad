//! Trains a character-level bigram language model on names — makemore's
//! opening act, before any MLP: one `[vocab, vocab]` table of logits
//! whose row `i` scores the character that follows token `i`.
//!
//! The whole model is a handful of recorded nodes: a `gather` picks the
//! context rows out of the table (the one-hot matmul), and
//! `cross_entropy` scores them against the next characters. The
//! gather's scatter-add gradient touches exactly the rows a batch
//! visits — the differentiable mirror of bigram counting. Minibatches
//! arrive as per-run feeds, so the tape never grows during training,
//! and sampling reads the trained table through the composite
//! `softmax`.
//!
//! Run with: `cargo run --release --example makemore_bigram`

mod corpus;

use poorgrad::{Network, Shape, Tensor, Tensorial, cross_entropy, init};

use corpus::{VOCABULARY_LEN, draw, from_token, load_names, shuffle, training_samples};

/// How many bigram pairs each training step feeds.
const BATCH_LEN: usize = 1024;

fn main() {
    let names = load_names();
    let mut samples = training_samples::<1>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);
    println!(
        "loaded {} names, {} bigram pairs",
        names.len(),
        samples.len()
    );

    let network = Network::new();
    let table = network.parameter(init::normal(7, 0.01)(&Shape::new([
        VOCABULARY_LEN,
        VOCABULARY_LEN,
    ])));

    // Contexts and targets are one-hot selections fed per run; the
    // defaults only fix the batch shape.
    let contexts = network.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));
    let targets = network.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));

    let logits = table.gather(contexts);
    let loss = cross_entropy(logits, targets);

    let table_symbol = table.symbol();
    let contexts_symbol = contexts.symbol();
    let targets_symbol = targets.symbol();
    let loss_symbol = loss.symbol();
    let recorded_nodes = network.len();

    // A fresh model is roughly uniform over the vocabulary, so the
    // first printed loss should sit near `ln(27) ~ 3.30`; the bigram
    // limit on this corpus is about `2.45`.
    let learning_rate = Tensor::new([], [10.0]);
    let mut network = network;
    for step in 0..1000 {
        let start = (step * BATCH_LEN) % (samples.len() - BATCH_LEN);
        let batch = &samples[start..start + BATCH_LEN];
        let batch_contexts: Vec<usize> = batch.iter().map(|&(context, _)| context[0]).collect();
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
        if step % 100 == 0 {
            println!(
                "step {step:4}: minibatch loss = {:.4}",
                evaluation.of(loss_value).to_vec()[0]
            );
        }
        let gradients = evaluation.backward(loss_value);
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    assert_eq!(network.len(), recorded_nodes);
    println!("the tape held {recorded_nodes} nodes through every step");

    // The trained logits exponentiate into transition probabilities
    // through the composite softmax: one more recorded expression.
    let probabilities = network.resolve(table_symbol).softmax(1);
    let evaluation = network.forward();
    let probabilities = evaluation
        .of(probabilities)
        .as_slice()
        .expect("a computed softmax is contiguous")
        .to_vec();

    println!("sampled names:");
    let mut state: u64 = 7;
    for _ in 0..10 {
        let mut token = 0;
        let mut name = String::new();
        loop {
            let row = &probabilities[token * VOCABULARY_LEN..(token + 1) * VOCABULARY_LEN];
            token = draw(row, &mut state);
            if token == 0 {
                break;
            }
            name.push(from_token(token));
        }
        println!("  {name}");
    }
}
