//! Trains a small MLP on XOR at tensor granularity, feeding minibatches
//! through a graph recorded once.
//!
//! A `[2, 4, 1]` perceptron records a handful of tensor nodes, the samples
//! arrive as per-run feeds instead of graph state, and every training step
//! binds a different minibatch with `forward_with` while the tape never grows.
//! Recording once and running anywhere covers the data, not just the targets —
//! the closing chart rasterizes the learned surface through the very same
//! two-row expression the training fed.
//!
//! Run with: `cargo run --example mlp_xor`

use malevich::{Cells, Frame, Line, Plot};
use poorgrad::{Mlp, Network, Tensor, Tensorial, init};

/// The resolution of the decision-surface chart: how many grid cells
/// span the unit square along each axis.
const SURFACE_COLUMNS: usize = 24;
const SURFACE_ROWS: usize = 12;

fn main() {
    let network = Network::new();
    // A deterministic seeded initializer keeps the run reproducible
    // while hidden-unit symmetry still breaks.
    let mlp = Mlp::new(&network, &[2, 4, 1], init::uniform(7, 0.5));

    // Declared inputs: the minibatch arrives per run, so the defaults
    // only fix the shapes — two samples of two features, two targets.
    let x = network.input(Tensor::filled([2, 2], 0.0_f32));
    let y = network.input(Tensor::filled([2, 1], 0.0));

    let predicted = mlp.express(&network, x);
    let error = predicted - y;
    let loss = (error * error).sum();

    let x_symbol = x.symbol();
    let y_symbol = y.symbol();
    let loss_symbol = loss.symbol();
    let predicted_symbol = predicted.symbol();
    let recorded_nodes = network.len();

    // The XOR truth table with targets in [-1, 1], split into two
    // minibatches that alternate across training steps.
    let minibatches = [
        (
            Tensor::new([2, 2], [0.0, 0.0, 0.0, 1.0]),
            Tensor::new([2, 1], [-1.0, 1.0]),
        ),
        (
            Tensor::new([2, 2], [1.0, 0.0, 1.0, 1.0]),
            Tensor::new([2, 1], [1.0, -1.0]),
        ),
    ];

    let learning_rate = Tensor::new([], [0.05]);
    let mut network = network;
    let mut losses = Vec::new();
    for step in 0..4000 {
        let (batch_x, batch_y) = &minibatches[step % minibatches.len()];
        let loss_value = network.resolve(loss_symbol);
        let evaluation =
            network.forward_with([(x_symbol, batch_x.clone()), (y_symbol, batch_y.clone())]);
        let batch_loss = evaluation.of(loss_value).to_vec()[0];
        losses.push(batch_loss);
        if step % 800 == 0 {
            println!("step {step:4}: minibatch loss = {batch_loss:.6}");
        }
        let gradients = evaluation.backward(loss_value);
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    assert_eq!(network.len(), recorded_nodes);
    println!("the tape held {recorded_nodes} nodes through every step");

    // On a log scale the descent's exponential tail is a straight
    // line; the two alternating minibatches braid around it.
    println!(
        "{}",
        Plot::new()
            .layer(Line::y(&losses[..]).label("minibatch"))
            .title("xor training")
            .x_label("step")
            .y_label("sum of squared errors")
            .log_y()
            .render(&Frame::detect())
    );

    println!("predictions (target in parentheses):");
    for (batch_x, batch_y) in &minibatches {
        let evaluation =
            network.forward_with([(x_symbol, batch_x.clone()), (y_symbol, batch_y.clone())]);
        let outputs = evaluation.of(network.resolve(predicted_symbol));
        for (sample, (prediction, target)) in outputs.iter().zip(batch_y.iter()).enumerate() {
            let features = batch_x.as_slice().expect("a fed minibatch is contiguous");
            let features = &features[sample * 2..sample * 2 + 2];
            println!(
                "  {:?} -> {prediction:+.3} ({target:+.0})",
                (features[0], features[1])
            );
        }
    }

    // The learned surface, sampled at grid cell centers over the unit
    // square through the same two-row expression training used, fed
    // two points per run.
    let centers: Vec<(f32, f32)> = (0..SURFACE_ROWS)
        .flat_map(|row| {
            (0..SURFACE_COLUMNS).map(move |column| {
                (
                    (column as f32 + 0.5) / SURFACE_COLUMNS as f32,
                    (row as f32 + 0.5) / SURFACE_ROWS as f32,
                )
            })
        })
        .collect();
    let mut surface = Vec::with_capacity(centers.len());
    for pair in centers.chunks(2) {
        let &[(x0, y0), (x1, y1)] = pair else {
            unreachable!("the even grid splits into exact pairs");
        };
        let evaluation = network.forward_with([(x_symbol, Tensor::new([2, 2], [x0, y0, x1, y1]))]);
        surface.extend(evaluation.of(network.resolve(predicted_symbol)).to_vec());
    }
    println!(
        "{}",
        Plot::new()
            .layer(Cells::matrix(SURFACE_COLUMNS, surface).extents((0.0, 1.0), (0.0, 1.0)))
            .colorbar()
            .title("the learned xor surface")
            .render(&Frame::detect())
    );
}
