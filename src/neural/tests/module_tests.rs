use crate::{Activation, Linear, Sequential, Tape, Tensor};

use super::{Module, named_parameters, parameters};

/// Builds a small tree exercising positional segments and stateless
/// stages.
fn tree(tape: &Tape<Tensor<f64>>) -> Sequential<Tensor<f64>> {
    let entry = Linear::new(
        tape,
        Tensor::filled([3, 4], 0.0_f64),
        Tensor::filled([4], 0.0),
    );
    let inner = Linear::new(
        tape,
        Tensor::filled([4, 4], 0.0_f64),
        Tensor::filled([4], 0.0),
    );
    Sequential::new()
        .then(entry)
        .then(Activation::Tanh)
        .then(inner)
}

#[test]
fn named_parameters_carry_dotted_paths() {
    let tape = Tape::new();
    let model = tree(&tape);
    let named = named_parameters(&model);
    let rendered: Vec<String> = named.iter().map(|(path, _)| path.to_string()).collect();
    // The activation contributes nothing, so the second linear stage
    // keeps its positional index.
    assert_eq!(rendered, ["0.weights", "0.bias", "2.weights", "2.bias"]);
}

#[test]
fn parameters_follow_visit_order() {
    let tape = Tape::new();
    let model = tree(&tape);
    let flat = parameters(&model);
    let named = named_parameters(&model);
    assert_eq!(flat.len(), 4);
    for (position, (_, symbol)) in named.iter().enumerate() {
        assert_eq!(flat[position], *symbol);
    }
}

#[test]
fn sequential_expresses_through_dyn_stages() {
    let tape = Tape::new();
    let model = tree(&tape);
    let input = tape.leaf(Tensor::filled([2, 3], 0.5_f64));
    let output = model.express(&tape, input).symbol();
    // Zero weights and biases: tanh(0) = 0, and the second affine
    // stage maps it to zero again, so the output is exactly zero.
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(output).to_vec(), vec![0.0; 8]);
}
