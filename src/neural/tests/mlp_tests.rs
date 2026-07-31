use crate::{Network, Shape, Tensor, Tensorial};

use super::Mlp;

/// Returns a deterministic initializer: an xorshift generator filling
/// each requested shape with small values, so tests are reproducible
/// while hidden-unit symmetry still breaks.
fn deterministic_initializer() -> impl FnMut(&Shape) -> Tensor<f64> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    move |shape| {
        let elements: Vec<f64> = (0..shape.volume())
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
            })
            .collect();
        Tensor::new(shape, elements)
    }
}

#[test]
fn new_builds_the_topology() {
    let network = Network::new();
    let mlp = Mlp::new(&network, &[3, 4, 4, 1], deterministic_initializer());
    // Three layers, each one weight tensor and one bias tensor.
    assert_eq!(mlp.parameters().count(), 6);
    assert_eq!(network.len(), 6);
}

#[test]
#[should_panic(expected = "input and an output width")]
fn new_rejects_degenerate_topologies() {
    let network = Network::<Tensor<f64>>::new();
    Mlp::new(&network, &[3], deterministic_initializer());
}

#[test]
fn express_chains_layers() {
    let network = Network::new();
    let mlp = Mlp::new(&network, &[3, 4, 2], deterministic_initializer());
    let input = network.leaf(Tensor::filled([5, 3], 0.5_f64));

    let output = mlp.express(&network, input);
    assert_eq!(output.shape(), Shape::new([5, 2]));
}

#[test]
fn mlp_learns_xor() {
    let network = Network::new();
    let mlp = Mlp::new(&network, &[2, 4, 1], deterministic_initializer());
    let x = network.leaf(Tensor::new(
        [4, 2],
        [0.0_f64, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
    ));
    let y = network.leaf(Tensor::new([4, 1], [-1.0, 1.0, 1.0, -1.0]));

    let predicted = mlp.express(&network, x);
    let error = predicted - y;
    let loss = (error * error).sum();
    let loss_symbol = loss.symbol();
    let predicted_symbol = predicted.symbol();

    let learning_rate = Tensor::new([], [0.05]);
    let mut network = network;
    for _ in 0..2000 {
        let loss = network.resolve(loss_symbol);
        let evaluation = network.forward();
        let gradients = evaluation.backward(loss);
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    let evaluation = network.forward();
    let outputs = evaluation.of(network.resolve(predicted_symbol));
    for (prediction, target) in outputs.iter().zip([-1.0, 1.0, 1.0, -1.0]) {
        assert!(
            (prediction - target).abs() < 0.2,
            "prediction {prediction} misses target {target}"
        );
    }
}
