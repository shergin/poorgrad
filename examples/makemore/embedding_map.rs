//! Trains the Bengio-style MLP with a two-dimensional character
//! embedding and draws the embedding space right in the terminal,
//! letters as `malevich` text marks — the terminal edition of the
//! classic makemore scatter plot.
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

use std::time::Instant;

use malevich::{Color, Frame, Plot, Text};
use poorgrad::{Mlp, Network, Shape, Tensor, Tensorial, cross_entropy, init};

use chart::loss_chart;
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

/// Spreads `points` so no two share a cell of a `columns` by `rows`
/// grid laid over their bounding box: a point whose cell is taken moves
/// to the nearest free cell within a small ring. A later text mark
/// would otherwise overwrite an earlier one where the space clusters —
/// which is exactly where the interesting letters are — so the chart
/// accepts a slight positional lie to keep every letter visible. A flat
/// axis widens to a unit span so a degenerate cloud still spreads.
fn spread(points: &mut [(f64, f64)], columns: usize, rows: usize) {
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    let mut y_min = f64::INFINITY;
    let mut y_max = f64::NEG_INFINITY;
    for &(x, y) in points.iter() {
        x_min = x_min.min(x);
        x_max = x_max.max(x);
        y_min = y_min.min(y);
        y_max = y_max.max(y);
    }
    let x_span = if x_max > x_min { x_max - x_min } else { 1.0 };
    let y_span = if y_max > y_min { y_max - y_min } else { 1.0 };

    let mut taken = vec![vec![false; columns]; rows];
    for point in points.iter_mut() {
        let column = ((point.0 - x_min) / x_span * (columns - 1) as f64).round() as isize;
        let row = ((point.1 - y_min) / y_span * (rows - 1) as f64).round() as isize;
        'placed: for radius in 0..=3_isize {
            for row_offset in -radius..=radius {
                for column_offset in -radius..=radius {
                    if row_offset.abs().max(column_offset.abs()) != radius {
                        continue;
                    }
                    let target_row = row + row_offset;
                    let target_column = column + column_offset;
                    if target_row < 0
                        || target_row >= rows as isize
                        || target_column < 0
                        || target_column >= columns as isize
                    {
                        continue;
                    }
                    if !taken[target_row as usize][target_column as usize] {
                        taken[target_row as usize][target_column as usize] = true;
                        point.0 = x_min + target_column as f64 / (columns - 1) as f64 * x_span;
                        point.1 = y_min + target_row as f64 / (rows - 1) as f64 * y_span;
                        break 'placed;
                    }
                }
            }
        }
    }
}

/// Renders the `[vocab, 2]` embedding `table` as a terminal map, one
/// text mark per token: vowels highlighted, the padding token dimmed,
/// consonants in the default foreground.
fn embedding_chart(title: &str, table: &Tensor<f32>) -> String {
    // The map wants more rows than the default third-of-terminal strip:
    // every letter is one cell, so vertical resolution is legibility.
    let mut frame = Frame::detect();
    frame.height = frame.height.max(24);

    let elements = table.to_vec();
    let mut points: Vec<(f64, f64)> = (0..VOCABULARY_LEN)
        .map(|token| {
            (
                f64::from(elements[token * EMBED_DIM]),
                f64::from(elements[token * EMBED_DIM + 1]),
            )
        })
        .collect();
    // The spread grid stands in for the plot area's cells, held a bit
    // coarser than the frame since malevich widens the domains to its
    // ticks and spends some cells on axis furniture.
    spread(
        &mut points,
        frame.width.saturating_sub(12).max(20),
        frame.height.saturating_sub(6).max(10),
    );

    // The domains pad one spread cell beyond the bounding box: a text
    // mark anchored exactly on the domain's right or bottom edge is
    // clipped away (malevich 1.11.1), and edge letters read better with
    // a margin anyway.
    let mut x_bounds = (f64::INFINITY, f64::NEG_INFINITY);
    let mut y_bounds = (f64::INFINITY, f64::NEG_INFINITY);
    for &(x, y) in &points {
        x_bounds = (x_bounds.0.min(x), x_bounds.1.max(x));
        y_bounds = (y_bounds.0.min(y), y_bounds.1.max(y));
    }
    let x_margin = (x_bounds.1 - x_bounds.0).max(1.0) / 40.0;
    let y_margin = (y_bounds.1 - y_bounds.0).max(1.0) / 16.0;

    let mut plot = Plot::new()
        .title(title)
        .x_domain(x_bounds.0 - x_margin, x_bounds.1 + x_margin)
        .y_domain(y_bounds.0 - y_margin, y_bounds.1 + y_margin);
    for (token, &(x, y)) in points.iter().enumerate() {
        let letter = from_token(token);
        let color = match letter {
            'a' | 'e' | 'i' | 'o' | 'u' => Color::BrightCyan,
            '.' => Color::BrightBlack,
            _ => Color::Default,
        };
        plot = plot.layer(Text::at(x, y, String::from(letter)).color(color));
    }
    plot.render_best(&frame)
}

fn main() {
    let names = load_names();
    let mut samples = training_samples::<CONTEXT_LEN>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);
    println!("loaded {} names, {} samples", names.len(), samples.len());

    let network: Network<Tensor<f32>> = Network::new();
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

    println!(
        "{}",
        embedding_chart(
            "embedding space before training (the seeded blob)",
            &embeddings.payload().unwrap(),
        )
    );

    // The two-dimensional bottleneck lands near 2.4 where ten
    // dimensions reach 2.25: the price of a plottable space.
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
        // Slice the run to the loss it reads.
        let run = network.forward_for(
            [loss_symbol],
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
        window_loss += batch_loss;
        if (step + 1) % 1000 == 0 {
            println!(
                "steps {:4}..{:4}: mean minibatch loss = {:.4}",
                step + 1 - 1000,
                step + 1,
                window_loss / 1000.0
            );
            window_loss = 0.0;
        }
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
    println!("{}", loss_chart("embedding map training", &losses));

    let table = network.resolve(embeddings_symbol).payload().unwrap();
    println!(
        "{}",
        embedding_chart(
            "embedding space after training (vowels highlighted)",
            &table
        )
    );
}
