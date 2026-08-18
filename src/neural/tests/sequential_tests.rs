use crate::{Activation, Linear, Module, Tape, Tensor};

use super::Sequential;

#[test]
fn an_empty_chain_is_the_identity() {
    let tape = Tape::new();
    let chain: Sequential<Tensor<f64>> = Sequential::new();
    assert!(chain.is_empty());
    let input = tape.leaf(Tensor::filled([2], 1.0_f64));
    let output = chain.express(&tape, input);
    // No stage records anything: the output is the input node itself.
    assert_eq!(output.symbol(), input.symbol());
}

#[test]
fn stages_chain_in_order() {
    let tape = Tape::new();
    // A negating affine stage before the relu: the chain order is
    // observable because relu(-x) differs from -relu(x).
    let negation = Linear::new(
        &tape,
        Tensor::new([2, 2], [-1.0_f64, 0.0, 0.0, -1.0]),
        Tensor::filled([2], 0.0),
    );
    let chain = Sequential::new().then(negation).then(Activation::Relu);
    assert_eq!(chain.len(), 2);
    let input = tape.leaf(Tensor::new([2, 2], [-1.0_f64, 2.0, -3.0, 4.0]));
    let output = chain.express(&tape, input).symbol();

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(output).to_vec(), vec![1.0, 0.0, 3.0, 0.0]);
}
