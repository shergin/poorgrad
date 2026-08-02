//! Trains the Bengio-style MLP with a two-dimensional character
//! embedding and draws the embedding space right in the terminal,
//! letters as marks — the terminal edition of the classic makemore
//! scatter plot.
//!
//! Two dimensions cost some loss against the ten-dimensional examples;
//! they buy a picture. The map prints twice — the seeded blob before
//! training and the organized space after — so the structure the
//! gradient carves out (watch the vowels drift together) is visible as
//! a before-and-after.
//!
//! Run with: `cargo run --release --example makemore_embedding_map`

mod chart;
#[allow(dead_code)]
mod corpus;

use poorgrad::{Mlp, Network, Shape, Tensor, Tensorial, cross_entropy, init};

use corpus::{VOCABULARY_LEN, from_token, load_names, shuffle, training_samples};

/// How many characters of history the model sees before predicting the
/// next one.
const CONTEXT_LEN: usize = 3;

/// How many dimensions the character embedding space has: two, so the
/// space is the page.
const EMBED_DIM: usize = 2;

/// How many neurons the tanh hidden layer has.
const HIDDEN_LEN: usize = 100;

/// How many samples each training step feeds.
const BATCH_LEN: usize = 64;

/// The chart size; twice as wide as tall, since terminal cells are
/// about twice as tall as they are wide.
const CHART_COLUMNS: usize = 72;
const CHART_ROWS: usize = 24;

/// Styles a token's letter for the chart: vowels highlighted, the
/// padding token dimmed, consonants plain.
fn styled(token: usize) -> String {
    let letter = from_token(token);
    match letter {
        'a' | 'e' | 'i' | 'o' | 'u' => format!("\x1b[1;36m{letter}\x1b[0m"),
        '.' => format!("\x1b[2m{letter}\x1b[0m"),
        _ => letter.to_string(),
    }
}

/// Renders the `[vocab, 2]` embedding `table` as a terminal scatter
/// chart, one letter per token.
fn embedding_chart(table: &Tensor<f64>) -> String {
    let elements = table.to_vec();
    let points: Vec<(f64, f64, String)> = (0..VOCABULARY_LEN)
        .map(|token| {
            (
                elements[token * EMBED_DIM],
                elements[token * EMBED_DIM + 1],
                styled(token),
            )
        })
        .collect();
    chart::scatter(&points, CHART_COLUMNS, CHART_ROWS)
}

fn main() {
    let names = load_names();
    let mut samples = training_samples::<CONTEXT_LEN>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);
    println!("loaded {} names, {} samples", names.len(), samples.len());

    let network = Network::new();
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
        vec![0; BATCH_LEN * CONTEXT_LEN],
        VOCABULARY_LEN,
        1.0,
    ));
    let targets = network.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));
    let embedded = embeddings
        .gather(contexts)
        .reshape([BATCH_LEN, CONTEXT_LEN * EMBED_DIM]);
    let loss = cross_entropy(mlp.express(&network, embedded), targets);

    let embeddings_symbol = embeddings.symbol();
    let contexts_symbol = contexts.symbol();
    let targets_symbol = targets.symbol();
    let loss_symbol = loss.symbol();

    println!("embedding space before training (the seeded blob):");
    println!("{}", embedding_chart(&embeddings.payload().unwrap()));

    // The two-dimensional bottleneck lands near 2.4 where ten
    // dimensions reach 2.25: the price of a plottable space.
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
        window_loss += evaluation.of(loss_value).to_vec()[0];
        if (step + 1) % 1000 == 0 {
            println!(
                "steps {:4}..{:4}: mean minibatch loss = {:.4}",
                step + 1 - 1000,
                step + 1,
                window_loss / 1000.0
            );
            window_loss = 0.0;
        }
        let gradients = evaluation.backward(loss_value);
        let learning_rate = if step < 4000 { &fast } else { &slow };
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    let table = network.resolve(embeddings_symbol).payload().unwrap();
    println!("embedding space after training (vowels highlighted):");
    println!("{}", embedding_chart(&table));
}
