//! Trains a linear model `w * x + b` with gradient descent, exercising one
//! shared network across threads.
//!
//! Two things run in parallel here. First, a single run of the shared
//! network feeds concurrent backward sweeps, one per target: runs are
//! per-thread state, the network is immutable. Second, several training
//! runs proceed simultaneously over one spec — each learning rate owns
//! its own `Parameters` state, and cloning state is all a what-if costs.
//!
//! Run with: `cargo run --example gradient_descent`

use std::time::Instant;

use rayon::prelude::*;

use malevich::{Frame, Line, Plot};
use topos::{Symbol, Tape};

fn main() {
    let tape = Tape::new();

    // Learnable parameters, starting from zero.
    let w = tape.parameter(0.0_f64);
    let b = tape.parameter(0.0);

    // Training data for the target line `y = 2 * x + 1`, recorded as plain
    // leaves. Each sample's squared error is kept as a separate target;
    // the total loss is their sum.
    let samples = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
    let mut sample_losses = Vec::new();
    for (x, y) in samples {
        let x = tape.leaf(x);
        let y = tape.leaf(y);
        let error = w * x + b - y;
        sample_losses.push(error * error);
    }
    // Values are `Copy`, so the per-sample losses fold into a total loss
    // with a plain reduce.
    let loss = sample_losses
        .iter()
        .copied()
        .reduce(|total, squared| total + squared)
        .expect("at least one sample");

    // Symbols are the names every phase after recording speaks; the
    // seal consumes the tape and hands back the immutable spec.
    let sample_losses: Vec<Symbol> = sample_losses.iter().map(|value| value.symbol()).collect();
    let (w, b, loss) = (w.symbol(), b.symbol(), loss.symbol());
    let network = tape.into_network();
    let parameters = network.parameters();

    // One run feeds many backward sweeps: each rayon thread
    // differentiates the same shared run for its own target.
    let run = network.forward(&parameters, []);
    let per_sample: Vec<f64> = sample_losses
        .par_iter()
        .map(|&sample_loss| {
            let gradients = run.backward(sample_loss);
            *gradients.of(w)
        })
        .collect();
    let total_gradient = *run.backward(loss).of(w);
    println!("per-sample d/dw, computed on separate threads: {per_sample:?}");
    println!(
        "their sum {} equals the total-loss d/dw {} by linearity",
        per_sample.iter().sum::<f64>(),
        total_gradient
    );

    // Parallel training: each learning rate clones the parameter state
    // and descends independently over the one shared spec, keeping its
    // whole loss history for the chart.
    let learning_rates = [0.005, 0.02, 0.05];
    let training = Instant::now();
    let runs: Vec<(f64, Vec<f64>, f64, f64)> = learning_rates
        .par_iter()
        .map(|&learning_rate| {
            let mut parameters = parameters.clone();
            let mut losses = Vec::with_capacity(501);
            for _ in 0..500 {
                let run = network.forward(&parameters, []);
                losses.push(*run.of(loss));
                let gradients = run.backward(loss);
                parameters = parameters.step(&gradients, |parameter, gradient| {
                    parameter - learning_rate * gradient
                });
            }
            let run = network.forward(&parameters, []);
            losses.push(*run.of(loss));
            (learning_rate, losses, *parameters.of(w), *parameters.of(b))
        })
        .collect();

    println!(
        "trained {} states of 500 steps in {:.3}s",
        learning_rates.len(),
        training.elapsed().as_secs_f64()
    );

    println!("parallel training on cloned states (target: w = 2, b = 1):");
    for (learning_rate, losses, w, b) in &runs {
        let final_loss = losses.last().expect("every run records its final loss");
        println!("  lr = {learning_rate:5.3}: loss = {final_loss:.6}, w = {w:.3}, b = {b:.3}");
    }

    // The same three descents as curves: on a log scale a constant
    // convergence rate is a straight line, so the slope is the rate.
    let mut plot = Plot::new()
        .title("gradient descent per learning rate")
        .x_label("step")
        .y_label("loss")
        .log_y();
    for (learning_rate, losses, ..) in &runs {
        plot = plot.layer(Line::y(&losses[..]).label(format!("lr = {learning_rate}")));
    }
    println!("{}", plot.render_best(&Frame::detect()));
}
