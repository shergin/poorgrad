use crate::{Network, Shape, Tensor, Tensorial};

use super::{Activation, Layer};

#[test]
fn new_allocates_weights_and_bias() {
    let network = Network::new();
    let layer = Layer::new(
        &network,
        Tensor::filled([3, 2], 0.0_f64),
        Tensor::filled([2], 0.0),
        Activation::Identity,
    );
    // One weight tensor and one bias tensor, regardless of size.
    assert_eq!(network.len(), 2);
    assert_eq!(layer.parameters().count(), 2);
}

#[test]
#[should_panic(expected = "must be rank 2")]
fn new_rejects_non_matrix_weights() {
    let network = Network::new();
    Layer::new(
        &network,
        Tensor::filled([3], 0.0_f64),
        Tensor::filled([2], 0.0),
        Activation::Identity,
    );
}

#[test]
#[should_panic(expected = "disagree on outputs")]
fn new_rejects_mismatched_bias() {
    let network = Network::new();
    Layer::new(
        &network,
        Tensor::filled([3, 2], 0.0_f64),
        Tensor::filled([3], 0.0),
        Activation::Identity,
    );
}

#[test]
fn express_records_tensor_granularity() {
    let network = Network::new();
    let layer = Layer::new(
        &network,
        Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]),
        Tensor::new([2], [10.0, 20.0]),
        Activation::Identity,
    );
    let input = network.leaf(Tensor::new([3, 2], [1.0, 0.0, 0.0, 1.0, 1.0, 1.0]));
    let nodes_before = network.len();

    let output = layer.express(&network, input);

    // The whole layer is three recorded nodes: the product, the bias
    // broadcast, and the shifted sum.
    assert_eq!(network.len(), nodes_before + 3);
    assert_eq!(output.shape(), Shape::new([3, 2]));

    let evaluation = network.forward();
    assert_eq!(
        evaluation.of(output).to_vec(),
        &[11.0, 22.0, 13.0, 24.0, 14.0, 26.0]
    );
}

#[test]
fn layer_trains_toward_targets() {
    // Fit `y = x . w + b` for `w = [[2], [-1]]` and `b = [0.5]`, feeding
    // the whole batch through one tensor-granularity layer.
    let network = Network::new();
    let layer = Layer::new(
        &network,
        Tensor::filled([2, 1], 0.0_f64),
        Tensor::filled([1], 0.0),
        Activation::Identity,
    );
    let x = network.leaf(Tensor::new(
        [4, 2],
        [1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0],
    ));
    let y = network.leaf(Tensor::new([4, 1], [2.5, -0.5, 1.5, 3.5]));

    let predicted = layer.express(&network, x);
    let error = predicted - y;
    let loss = (error * error).sum();
    let loss_symbol = loss.symbol();

    let learning_rate = Tensor::new([], [0.05]);
    let mut network = network;
    for _ in 0..300 {
        let loss = network.resolve(loss_symbol);
        let evaluation = network.forward();
        let gradients = evaluation.backward(loss);
        network = network.updated(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    let parameters: Vec<_> = layer.parameters().collect();
    let weights = network.resolve(parameters[0]).payload().unwrap();
    let bias = network.resolve(parameters[1]).payload().unwrap();
    assert!((weights.to_vec()[0] - 2.0).abs() < 1e-3);
    assert!((weights.to_vec()[1] + 1.0).abs() < 1e-3);
    assert!((bias.to_vec()[0] - 0.5).abs() < 1e-3);
}
