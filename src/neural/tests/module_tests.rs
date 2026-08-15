use crate::{Activation, Linear, Network, Residual, Sequential, Tensor};

use super::{Module, named_parameters, parameters};

/// Builds a small tree exercising positional segments, stateless
/// stages, and `Residual` transparency.
fn tree(network: &Network<Tensor<f64>>) -> Sequential<Tensor<f64>> {
    let entry = Linear::new(
        network,
        Tensor::filled([3, 4], 0.0_f64),
        Tensor::filled([4], 0.0),
    );
    let inner = Linear::new(
        network,
        Tensor::filled([4, 4], 0.0_f64),
        Tensor::filled([4], 0.0),
    );
    Sequential::new()
        .then(entry)
        .then(Activation::Tanh)
        .then(Residual(inner))
}

#[test]
fn named_parameters_carry_dotted_paths() {
    let network = Network::new();
    let model = tree(&network);
    let named = named_parameters(&model);
    let rendered: Vec<String> = named.iter().map(|(path, _)| path.to_string()).collect();
    // The activation contributes nothing, and `Residual` is
    // path-transparent: its inner parameters keep the stage's index.
    assert_eq!(rendered, ["0.weights", "0.bias", "2.weights", "2.bias"]);
}

#[test]
fn parameters_follow_visit_order() {
    let network = Network::new();
    let model = tree(&network);
    let flat = parameters(&model);
    let named = named_parameters(&model);
    assert_eq!(flat.len(), 4);
    for (position, (_, symbol)) in named.iter().enumerate() {
        assert_eq!(flat[position], *symbol);
    }
}

#[test]
fn sequential_expresses_through_dyn_stages() {
    let network = Network::new();
    let model = tree(&network);
    let input = network.leaf(Tensor::filled([2, 3], 0.5_f64));
    let output = model.express(&network, input);
    // Zero weights and biases: tanh(0) = 0, and the residual passes
    // the zero through, so the output is exactly zero.
    let run = network.forward();
    assert_eq!(run.of(output).to_vec(), vec![0.0; 8]);
}
