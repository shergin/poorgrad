//! Trains an MLP classifier on two interleaved half-moons and draws
//! the learned decision surface — the classic playground picture for a
//! small neural network, as a before-and-after: the scattered moons
//! first, the surface that separates them at the end, both charts
//! pinned to the same domains so they align.
//!
//! The tape holds two expressions of the same parameters: a
//! batch-shaped one for training and a grid-shaped twin whose input
//! defaults to the chart raster, so one plain `forward` at the end
//! rasterizes the decision function.
//!
//! Run with: `cargo run --release --example moons`

use std::f32::consts::PI;
use std::time::Instant;

use malevich::{Cells, Frame, Line, Plot, Points};
use topos::{Mlp, Shape, Tape, Tensor, Tensorial, init};

/// How many points each half-moon holds.
const MOON_LEN: usize = 100;

/// The standard deviation of the noise scattering the moons.
const NOISE_DEVIATION: f64 = 0.1;

/// The resolution of the decision-surface chart grid.
const SURFACE_COLUMNS: usize = 48;
const SURFACE_ROWS: usize = 16;

/// The data window the surface rasterizes: both moons with a margin.
const X_SPAN: (f32, f32) = (-1.5, 2.5);
const Y_SPAN: (f32, f32) = (-1.0, 1.5);

/// How many full-batch gradient descent steps the training takes.
const STEP_COUNT: usize = 3000;

/// Builds one half-moon of `MOON_LEN` noisy points: the upper arch
/// when `flipped` is false, the interleaved lower arch when true.
fn moon(flipped: bool, noise: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut moon_x = Vec::with_capacity(MOON_LEN);
    let mut moon_y = Vec::with_capacity(MOON_LEN);
    for index in 0..MOON_LEN {
        let angle = PI * index as f32 / (MOON_LEN - 1) as f32;
        let (x, y) = if flipped {
            (1.0 - angle.cos(), 0.5 - angle.sin())
        } else {
            (angle.cos(), angle.sin())
        };
        moon_x.push(x + noise[index * 2]);
        moon_y.push(y + noise[index * 2 + 1]);
    }
    (moon_x, moon_y)
}

fn main() {
    let noise: Tensor<f32> = init::normal(5, NOISE_DEVIATION)(&Shape::new([2 * MOON_LEN, 2]));
    let noise = noise.to_vec();
    let (upper_x, upper_y) = moon(false, &noise[..2 * MOON_LEN]);
    let (lower_x, lower_y) = moon(true, &noise[2 * MOON_LEN..]);

    // The batch interleaves each point's coordinates row by row, the
    // upper moon labeled `+1` and the lower `-1`.
    let mut feature_values = Vec::with_capacity(4 * MOON_LEN);
    for (x, y) in upper_x
        .iter()
        .zip(&upper_y)
        .chain(lower_x.iter().zip(&lower_y))
    {
        feature_values.push(*x);
        feature_values.push(*y);
    }
    let mut target_values = vec![1.0_f32; MOON_LEN];
    target_values.extend(vec![-1.0; MOON_LEN]);

    let tape: Tape<Tensor<f32>> = Tape::new();
    let mlp = Mlp::new(&tape, &[2, 16, 16, 1], init::xavier(7));

    let input = tape.input(Tensor::new([2 * MOON_LEN, 2], feature_values));
    let expected = tape.input(Tensor::new([2 * MOON_LEN, 1], target_values.clone()));
    let predicted = mlp.express(&tape, input);
    let error = predicted - expected;
    let loss = (error * error).sum();

    // The rasterizing twin: the same parameters expressed over the
    // chart grid's cell centers, recorded once next to the training
    // expression.
    let mut surface_centers = Vec::with_capacity(2 * SURFACE_COLUMNS * SURFACE_ROWS);
    for row in 0..SURFACE_ROWS {
        for column in 0..SURFACE_COLUMNS {
            let fraction_x = (column as f32 + 0.5) / SURFACE_COLUMNS as f32;
            let fraction_y = (row as f32 + 0.5) / SURFACE_ROWS as f32;
            surface_centers.push(X_SPAN.0 + fraction_x * (X_SPAN.1 - X_SPAN.0));
            surface_centers.push(Y_SPAN.0 + fraction_y * (Y_SPAN.1 - Y_SPAN.0));
        }
    }
    let surface_input = tape.input(Tensor::new(
        [SURFACE_COLUMNS * SURFACE_ROWS, 2],
        surface_centers,
    ));
    let surface_predicted = mlp.express(&tape, surface_input);

    let (predicted, surface_predicted, loss) = (
        predicted.symbol(),
        surface_predicted.symbol(),
        loss.symbol(),
    );
    let network = tape.into_network();
    let mut parameters = network.parameters();

    println!(
        "{}",
        Plot::new()
            .layer(Points::xy(&upper_x[..], &upper_y[..]).label("class +1"))
            .layer(Points::xy(&lower_x[..], &lower_y[..]).label("class -1"))
            .x_domain(f64::from(X_SPAN.0), f64::from(X_SPAN.1))
            .y_domain(f64::from(Y_SPAN.0), f64::from(Y_SPAN.1))
            .title("the two moons")
            .render_best(&Frame::detect())
    );

    let learning_rate = Tensor::new([], [0.0003]);
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
            .title("two moons training")
            .x_label("step")
            .y_label("sum of squared errors")
            .render_best(&Frame::detect())
    );

    // One forward serves both readouts: the training points' signs for
    // the accuracy line and the grid twin for the surface chart.
    let run = network.forward(&parameters, []);
    let classified = run
        .of(predicted)
        .to_vec()
        .iter()
        .zip(&target_values)
        .filter(|(prediction, target)| prediction.signum() == target.signum())
        .count();
    println!(
        "classified {classified} of {} training points",
        2 * MOON_LEN
    );

    let surface = run.of(surface_predicted).to_vec();
    println!(
        "{}",
        Plot::new()
            .layer(Cells::matrix(SURFACE_COLUMNS, surface).extents(
                (f64::from(X_SPAN.0), f64::from(X_SPAN.1)),
                (f64::from(Y_SPAN.0), f64::from(Y_SPAN.1)),
            ))
            .colorbar()
            .title("the learned decision surface")
            .render_best(&Frame::detect())
    );
}
