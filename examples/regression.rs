//! Fits a one-hidden-layer MLP to a noisy sine wave and draws the fit:
//! the sampled points, the curve they came from, and the curve the
//! model learned, in one chart.
//!
//! The graph is recorded once over a `[SAMPLE_LEN, 1]` input whose
//! default feed is the training set, so training is a plain `forward`
//! per step; charting feeds an evenly spaced grid through the same
//! expression, so the fit line is the model itself, not an
//! interpolation of its training outputs.
//!
//! Run with: `cargo run --release --example regression`

use std::time::Instant;

use malevich::{Frame, Line, Plot, Points};
use topos::{Mlp, Shape, Tape, Tensor, Tensorial, init};

/// How many noisy samples the model trains on; the chart grid reuses
/// the count so the recorded expression serves both.
const SAMPLE_LEN: usize = 96;

/// The half-width of the sampled domain, centered on zero.
const DOMAIN: f64 = 3.0;

/// The standard deviation of the noise added to the sampled curve.
const NOISE_DEVIATION: f64 = 0.1;

/// How many neurons the tanh hidden layer has.
const HIDDEN_LEN: usize = 16;

/// How many full-batch gradient descent steps the fit takes.
const STEP_COUNT: usize = 3000;

fn main() {
    // The dataset: x uniform over the domain and y = sin x plus noise,
    // both drawn from the same seeded initializers the parameters use.
    let features: Tensor<f32> = init::uniform(3, DOMAIN)(&Shape::new([SAMPLE_LEN, 1]));
    let noise: Tensor<f32> = init::normal(5, NOISE_DEVIATION)(&Shape::new([SAMPLE_LEN, 1]));
    let feature_values = features.to_vec();
    let target_values: Vec<f32> = feature_values
        .iter()
        .zip(noise.to_vec())
        .map(|(&x, noise)| x.sin() + noise)
        .collect();

    let tape: Tape<Tensor<f32>> = Tape::new();
    let mlp = Mlp::new(&tape, &[1, HIDDEN_LEN, 1], init::xavier(7));

    let input = tape.input(features);
    let expected = tape.input(Tensor::new([SAMPLE_LEN, 1], target_values.clone()));
    let predicted = mlp.express(&tape, input);
    let error = predicted - expected;
    let loss = (error * error).sum();

    let (input, predicted, loss) = (input.symbol(), predicted.symbol(), loss.symbol());
    let network = tape.into_network();
    let mut parameters = network.parameters();

    let learning_rate = Tensor::new([], [0.001]);
    let mut losses = Vec::new();
    let training = Instant::now();
    for step in 0..STEP_COUNT {
        let run = network.forward(&parameters, []);
        let batch_loss = run.of(loss).to_vec()[0];
        losses.push(batch_loss);
        if step % (STEP_COUNT / 5) == 0 {
            println!("step {step:4}: loss = {batch_loss:.4}");
        }
        let gradients = run.backward(loss);
        parameters = parameters.step(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    println!(
        "trained {} steps in {:.3}s",
        losses.len(),
        training.elapsed().as_secs_f64()
    );

    println!(
        "{}",
        Plot::new()
            .layer(Line::y(&losses[..]).label("full batch"))
            .title("regression training")
            .x_label("step")
            .y_label("sum of squared errors")
            .render_best(&Frame::detect())
    );

    // The fit, read over an evenly spaced grid fed through the trained
    // expression in place of the training samples.
    let grid: Vec<f32> = (0..SAMPLE_LEN)
        .map(|index| (((index as f64 / (SAMPLE_LEN - 1) as f64) * 2.0 - 1.0) * DOMAIN) as f32)
        .collect();
    let run = network.forward(
        &parameters,
        [(input, Tensor::new([SAMPLE_LEN, 1], grid.clone()))],
    );
    let fit = run.of(predicted).to_vec();

    println!(
        "{}",
        Plot::new()
            .layer(Points::xy(&feature_values[..], &target_values[..]).label("samples"))
            .layer(Line::function(-DOMAIN..DOMAIN, f64::sin).label("sin x"))
            .layer(Line::xy(&grid[..], &fit[..]).label("fit"))
            .title("a tanh mlp fit to noisy sin x")
            .x_label("x")
            .render_best(&Frame::detect())
    );
}
