//! Trains a linear model `w * x + b` with gradient descent, exercising one
//! shared network across threads.
//!
//! Two things run in parallel here. First, a single evaluation of the
//! shared network feeds concurrent backward sweeps, one per target: runs
//! are per-thread state, the network is never mutated. Second, several
//! training runs proceed simultaneously, each on its own O(1) fork of the
//! same recorded graph, one per learning rate.
//!
//! Run with: `cargo run --example gradient_descent`

use rayon::prelude::*;

use malevich::{Frame, Line, Plot};
use poorgrad::Network;

fn main() {
    let network = Network::new();

    // Learnable parameters, starting from zero.
    let w = network.parameter(0.0_f64);
    let b = network.parameter(0.0);

    // Training data for the target line `y = 2 * x + 1`, recorded as plain
    // leaves. Each sample's squared error is kept as a separate target;
    // the total loss is their sum.
    let samples = [(1.0, 3.0), (2.0, 5.0), (3.0, 7.0)];
    let mut sample_losses = Vec::new();
    for (x, y) in samples {
        let x = network.leaf(x);
        let y = network.leaf(y);
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

    // One evaluation feeds many backward sweeps: each rayon thread
    // differentiates the same shared network for its own target.
    let evaluation = network.forward();
    let per_sample: Vec<f64> = sample_losses
        .par_iter()
        .map(|&sample_loss| {
            let gradients = evaluation.backward(sample_loss);
            *gradients.of(w)
        })
        .collect();
    let total_gradient = *evaluation.backward(loss).of(w);
    println!("per-sample d/dw, computed on separate threads: {per_sample:?}");
    println!(
        "their sum {} equals the total-loss d/dw {} by linearity",
        per_sample.iter().sum::<f64>(),
        total_gradient
    );

    // Symbols survive generations and cross threads freely; every training
    // run below resolves them against its own generations.
    let w_symbol = w.symbol();
    let b_symbol = b.symbol();
    let loss_symbol = loss.symbol();

    // Parallel training: each learning rate gets an O(1) fork of the same
    // recorded graph and descends independently, keeping its whole loss
    // history for the chart.
    let learning_rates = [0.005, 0.02, 0.05];
    let runs: Vec<(f64, Vec<f64>, f64, f64)> = learning_rates
        .par_iter()
        .map(|&learning_rate| {
            let mut network = network.clone();
            let mut losses = Vec::with_capacity(501);
            for _ in 0..500 {
                let loss = network.resolve(loss_symbol);
                let evaluation = network.forward();
                losses.push(*evaluation.of(loss));
                let gradients = evaluation.backward(loss);
                network = network.update(&gradients, |parameter, gradient| {
                    parameter - learning_rate * gradient
                });
            }
            let loss = network.resolve(loss_symbol);
            let evaluation = network.forward();
            losses.push(*evaluation.of(loss));
            let w = network.resolve(w_symbol);
            let b = network.resolve(b_symbol);
            let w = w.payload().expect("parameters carry payloads");
            let b = b.payload().expect("parameters carry payloads");
            (learning_rate, losses, w, b)
        })
        .collect();

    println!("parallel training on forks (target: w = 2, b = 1):");
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
    println!("{}", plot.render(&Frame::detect()));
}
