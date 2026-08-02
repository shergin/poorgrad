//! Trains the same character-level MLP as `makemore_mlp_facade`, data
//! parallel: every step splits its minibatch into equal shards, runs
//! one forward and backward pass per shard concurrently on the shared
//! network, and averages the shard gradients into a single update.
//!
//! This leans on the engine's concurrency contract: runs never mutate
//! the network, feeds are run state rather than graph state, and
//! `Gradients` is a `Field`, so gradients from concurrent runs on one
//! generation combine with the field algebra. Because `cross_entropy`
//! normalizes each shard by its own mass, the average of equal-sized
//! shard gradients equals the full-batch gradient exactly, and summing
//! the shards in a fixed pairwise tree keeps the run deterministic
//! regardless of thread scheduling.
//!
//! Run with: `cargo run --release --example makemore_mlp_parallel`

mod corpus;

use std::time::Instant;

use rayon::prelude::*;

use poorgrad::{Gradients, Mlp, Network, Shape, Tensor, Tensorial, cross_entropy, init};

use corpus::{VOCABULARY_LEN, draw, from_token, load_names, shuffle, training_samples};

/// How many characters of history the model sees before predicting the
/// next one.
const CONTEXT_LEN: usize = 3;

/// How many dimensions the character embedding space has.
const EMBED_DIM: usize = 10;

/// How many neurons the tanh hidden layer has.
const HIDDEN_LEN: usize = 100;

/// How many concurrent shards each training step fans out.
///
/// The count is fixed rather than detected — the shard partition
/// decides the arithmetic, so a detected count would make runs differ
/// across machines. Rayon still schedules the fixed shards over every
/// core it has. The count trades per-run fixed cost (favoring fewer,
/// larger shards) against load balancing over heterogeneous cores
/// (favoring more, smaller ones); eight-by-eight measured fastest on an
/// eight-performance-core machine, where sixteen-by-four raised
/// utilization but paid more in run overhead than it recovered.
const SHARD_COUNT: usize = 8;

/// How many samples each shard carries; the effective batch is
/// `SHARD_COUNT * SHARD_LEN`, matching the serial examples' 64.
const SHARD_LEN: usize = 8;

/// Sums shard gradients as a pairwise tree whose shape depends only on
/// the shard count: the reduction runs its pairs concurrently and
/// finishes in logarithmic depth, while the tree — not the scheduler —
/// decides the order of additions, keeping the result deterministic.
fn tree_sum(mut layer: Vec<Gradients<Tensor<f64>>>) -> Gradients<Tensor<f64>> {
    while layer.len() > 1 {
        layer = layer
            .par_chunks(2)
            .map(|pair| match pair {
                [left, right] => left + right,
                [single] => single.clone(),
                _ => unreachable!("chunks of two hold one or two fields"),
            })
            .collect();
    }
    layer.into_iter().next().expect("at least one shard ran")
}

fn main() {
    let names = load_names();
    let mut samples = training_samples::<CONTEXT_LEN>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);
    println!("loaded {} names, {} samples", names.len(), samples.len());

    let network = Network::new();

    // The same model as the serial examples, recorded at shard shape:
    // the batch size is baked into the graph, so the parallel plan is
    // one shard-shaped expression run once per shard, not a wider one.
    let embeddings = network.parameter(init::normal(8, 1.0)(&Shape::new([
        VOCABULARY_LEN,
        EMBED_DIM,
    ])));
    let mlp = Mlp::new(
        &network,
        &[CONTEXT_LEN * EMBED_DIM, HIDDEN_LEN, VOCABULARY_LEN],
        init::xavier(7),
    );

    let contexts = network.input(Tensor::selection(
        vec![0; SHARD_LEN * CONTEXT_LEN],
        VOCABULARY_LEN,
        1.0,
    ));
    let targets = network.input(Tensor::selection(vec![0; SHARD_LEN], VOCABULARY_LEN, 1.0));
    let embedded = embeddings
        .gather(contexts)
        .reshape([SHARD_LEN, CONTEXT_LEN * EMBED_DIM]);
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

    let batch_len = SHARD_COUNT * SHARD_LEN;
    let shard_inverse = Tensor::new([], [1.0 / SHARD_COUNT as f64]);
    let fast = Tensor::new([], [0.1]);
    let slow = Tensor::new([], [0.01]);
    let mut network = network;
    let mut window_loss = 0.0;
    let training = Instant::now();
    for step in 0..5000 {
        let start = (step * batch_len) % (samples.len() - batch_len);
        let batch = &samples[start..start + batch_len];

        // Fan out: one immutable forward and backward run per shard,
        // all reading the same generation.
        let shard_results: Vec<(f64, Gradients<Tensor<f64>>)> = (0..SHARD_COUNT)
            .into_par_iter()
            .map(|shard| {
                let rows = &batch[shard * SHARD_LEN..(shard + 1) * SHARD_LEN];
                let shard_contexts: Vec<usize> = rows
                    .iter()
                    .flat_map(|(context, _)| context.iter().copied())
                    .collect();
                let shard_targets: Vec<usize> = rows.iter().map(|&(_, next)| next).collect();

                let loss_value = network.resolve(loss_symbol);
                let evaluation = network.forward_with([
                    (
                        contexts_symbol,
                        Tensor::selection(shard_contexts, VOCABULARY_LEN, 1.0),
                    ),
                    (
                        targets_symbol,
                        Tensor::selection(shard_targets, VOCABULARY_LEN, 1.0),
                    ),
                ]);
                let shard_loss = evaluation.of(loss_value).to_vec()[0];
                (shard_loss, evaluation.backward(loss_value))
            })
            .collect();

        let (shard_losses, shard_gradients): (Vec<f64>, Vec<Gradients<Tensor<f64>>>) =
            shard_results.into_iter().unzip();
        let batch_loss = shard_losses.iter().sum::<f64>() / SHARD_COUNT as f64;
        let gradients = tree_sum(shard_gradients)
            .map(|gradient| gradient.clone() * shard_inverse.broadcast_like(gradient));

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
        let learning_rate = if step < 4000 { &fast } else { &slow };
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }
    println!(
        "trained on {SHARD_COUNT} shards in {:.1}s",
        training.elapsed().as_secs_f64()
    );

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
