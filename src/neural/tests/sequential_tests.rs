use crate::{Activation, Flatten, MaxPool, Module, Network, Reshape, Tensor, max_pool};

use super::Sequential;

#[test]
fn an_empty_chain_is_the_identity() {
    let network = Network::new();
    let chain: Sequential<Tensor<f64>> = Sequential::new();
    assert!(chain.is_empty());
    let input = network.leaf(Tensor::filled([2], 1.0_f64));
    let output = chain.express(&network, input);
    // No stage records anything: the output is the input node itself.
    assert_eq!(output.symbol(), input.symbol());
}

#[test]
fn stages_chain_in_order() {
    let network = Network::new();
    let chain = Sequential::new()
        .then(Reshape::new([4]))
        .then(Activation::Relu);
    assert_eq!(chain.len(), 2);
    let input = network.leaf(Tensor::new([2, 2], [-1.0_f64, 2.0, -3.0, 4.0]));
    let output = chain.express(&network, input);

    let run = network.forward();
    assert_eq!(run.of(output).to_vec(), vec![0.0, 2.0, 0.0, 4.0]);
}

#[test]
fn flatten_collapses_after_the_batch_axis() {
    let network = Network::new();
    let input = network.leaf(Tensor::filled([2, 3, 4], 1.0_f64));
    let output = Flatten.express(&network, input);
    assert_eq!(output.shape().axes(), [2, 12]);
}

#[test]
fn pooling_modules_match_their_free_functions() {
    let network = Network::new();
    let elements: Vec<f64> = (0..16).map(|value| value as f64).collect();
    let image = Tensor::new([1, 1, 4, 4], elements);
    let input = network.leaf(image);

    let through_module = MaxPool::new(2, 2).express(&network, input);
    let through_function = max_pool(input, 2, 2);

    let run = network.forward();
    assert_eq!(
        run.of(through_module).to_vec(),
        run.of(through_function).to_vec()
    );
}
