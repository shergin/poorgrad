use crate::{Module, Network, Tensor};

use super::Linear;

#[test]
fn new_allocates_weights_and_bias() {
    let network = Network::new();
    let linear = Linear::new(
        &network,
        Tensor::filled([3, 2], 0.0_f64),
        Tensor::filled([2], 0.0),
    );
    // One weight tensor and one bias tensor, regardless of size.
    assert_eq!(network.len(), 2);
    assert_eq!(linear.parameters().count(), 2);
}

#[test]
#[should_panic(expected = "must be rank 2")]
fn new_rejects_non_matrix_weights() {
    let network = Network::new();
    Linear::new(
        &network,
        Tensor::filled([3], 0.0_f64),
        Tensor::filled([2], 0.0),
    );
}

#[test]
#[should_panic(expected = "disagree on outputs")]
fn new_rejects_mismatched_bias() {
    let network = Network::new();
    Linear::new(
        &network,
        Tensor::filled([3, 2], 0.0_f64),
        Tensor::filled([3], 0.0),
    );
}

#[test]
fn express_records_the_affine_transform() {
    let network = Network::new();
    let linear = Linear::new(
        &network,
        Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]),
        Tensor::new([2], [10.0, 20.0]),
    );
    let input = network.leaf(Tensor::new([1, 2], [1.0_f64, 1.0]));
    let output = linear.express(&network, input);

    let evaluation = network.forward();
    // [1, 1] x [[1, 2], [3, 4]] + [10, 20] = [14, 26]: affine alone,
    // no bundled activation.
    assert_eq!(evaluation.of(output).to_vec(), vec![14.0, 26.0]);
}

#[test]
fn from_symbols_ties_existing_parameters() {
    let network = Network::new();
    let original = Linear::new(
        &network,
        Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]),
        Tensor::new([2], [0.0, 0.0]),
    );
    let tied = Linear::from_symbols(original.weights(), original.bias());
    assert_eq!(network.len(), 2, "tying allocates nothing");

    let input = network.leaf(Tensor::new([1, 2], [1.0_f64, 1.0]));
    let first = original.express(&network, input);
    let second = tied.express(&network, input);
    let evaluation = network.forward();
    assert_eq!(
        evaluation.of(first).to_vec(),
        evaluation.of(second).to_vec()
    );
}
