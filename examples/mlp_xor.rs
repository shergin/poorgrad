//! Trains a small MLP on XOR at tensor granularity, feeding minibatches
//! through a graph recorded once.
//!
//! A `[2, 4, 1]` perceptron records a handful of tensor nodes, the samples
//! arrive as per-run feeds instead of graph state, and every training step
//! binds a different minibatch with `forward_with` while the tape never grows.
//! Recording once and running anywhere covers the data, not just the targets.
//!
//! Run with: `cargo run --example mlp_xor`

use poorgrad::{Mlp, Network, Tensor, Tensorial, init};

fn main() {
    let network = Network::new();
    // A deterministic seeded initializer keeps the run reproducible
    // while hidden-unit symmetry still breaks.
    let mlp = Mlp::new(&network, &[2, 4, 1], init::uniform(7, 0.5));

    // Declared inputs: the minibatch arrives per run, so the defaults
    // only fix the shapes — two samples of two features, two targets.
    let x = network.input(Tensor::filled([2, 2], 0.0_f64));
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
    for step in 0..4000 {
        let (batch_x, batch_y) = &minibatches[step % minibatches.len()];
        let loss_value = network.resolve(loss_symbol);
        let evaluation =
            network.forward_with([(x_symbol, batch_x.clone()), (y_symbol, batch_y.clone())]);
        if step % 800 == 0 {
            println!(
                "step {step:4}: minibatch loss = {:.6}",
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
}
