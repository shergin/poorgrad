use crate::{Module, Tape, Tensor};

use super::Linear;

#[test]
fn new_allocates_weights_and_bias() {
    let tape = Tape::new();
    let linear = Linear::new(
        &tape,
        Tensor::filled([3, 2], 0.0_f64),
        Tensor::filled([2], 0.0),
    );
    // One weight tensor and one bias tensor, regardless of size.
    assert_eq!(tape.len(), 2);
    assert_eq!(linear.parameters().count(), 2);
}

#[test]
#[should_panic(expected = "must be rank 2")]
fn new_rejects_non_matrix_weights() {
    let tape = Tape::new();
    Linear::new(
        &tape,
        Tensor::filled([3], 0.0_f64),
        Tensor::filled([2], 0.0),
    );
}

#[test]
#[should_panic(expected = "disagree on outputs")]
fn new_rejects_mismatched_bias() {
    let tape = Tape::new();
    Linear::new(
        &tape,
        Tensor::filled([3, 2], 0.0_f64),
        Tensor::filled([3], 0.0),
    );
}

#[test]
fn express_records_the_affine_transform() {
    let tape = Tape::new();
    let linear = Linear::new(
        &tape,
        Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]),
        Tensor::new([2], [10.0, 20.0]),
    );
    let input = tape.leaf(Tensor::new([1, 2], [1.0_f64, 1.0]));
    let output = linear.express(&tape, input).symbol();

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    // [1, 1] x [[1, 2], [3, 4]] + [10, 20] = [14, 26]: affine alone,
    // no bundled activation.
    assert_eq!(run.of(output).to_vec(), vec![14.0, 26.0]);
}

#[test]
fn from_symbols_ties_existing_parameters() {
    let tape = Tape::new();
    let original = Linear::new(
        &tape,
        Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]),
        Tensor::new([2], [0.0, 0.0]),
    );
    let tied = Linear::from_symbols(original.weights(), original.bias());
    assert_eq!(tape.len(), 2, "tying allocates nothing");

    let input = tape.leaf(Tensor::new([1, 2], [1.0_f64, 1.0]));
    let first = original.express(&tape, input).symbol();
    let second = tied.express(&tape, input).symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(first).to_vec(), run.of(second).to_vec());
}
