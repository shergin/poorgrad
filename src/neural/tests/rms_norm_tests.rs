use crate::{Shape, Tape, Tensor};

use super::RmsNorm;

#[test]
fn new_allocates_scale_and_epsilon() {
    let tape = Tape::new();
    let norm = RmsNorm::new(
        &tape,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([], 1e-5),
    );
    // One parameter and the epsilon constant, regardless of size.
    assert_eq!(tape.len(), 2);
    assert_eq!(norm.parameters().count(), 1);
}

#[test]
#[should_panic(expected = "must be rank 1")]
fn new_rejects_non_vector_scale() {
    let tape = Tape::new();
    RmsNorm::new(
        &tape,
        Tensor::filled([2, 2], 1.0_f64),
        Tensor::filled([], 1e-5),
    );
}

#[test]
#[should_panic(expected = "single value")]
fn new_rejects_multi_value_epsilon() {
    let tape = Tape::new();
    RmsNorm::new(
        &tape,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 1e-5),
    );
}

#[test]
fn express_normalizes_by_the_root_mean_square() {
    let tape = Tape::new();
    let norm = RmsNorm::new(&tape, Tensor::filled([2], 1.0_f64), Tensor::filled([], 0.0));
    // Sample rows `[2, 2]` and `[3, -3]`: mean squares `[4, 9]`, so the
    // roots are `[2, 3]` and the rows normalize to `[1, 1]` and
    // `[1, -1]`. No centering: the second row's sign survives.
    let input = tape.leaf(Tensor::new([2, 2], [2.0, 2.0, 3.0, -3.0]));

    let output = norm.express(&tape, input);
    assert_eq!(output.shape(), Shape::new([2, 2]));

    let output = output.symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(output).to_vec(), &[1.0, 1.0, 1.0, -1.0]);
}

#[test]
fn express_applies_the_learned_scale() {
    let tape = Tape::new();
    let norm = RmsNorm::new(
        &tape,
        Tensor::new([2], [2.0_f64, 5.0]),
        Tensor::filled([], 0.0),
    );
    let input = tape.leaf(Tensor::new([2, 2], [2.0, 2.0, 3.0, -3.0]));

    let output = norm.express(&tape, input).symbol();

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(output).to_vec(), &[2.0, 5.0, 2.0, -5.0]);
}

#[test]
fn express_records_tensor_granularity() {
    let tape = Tape::new();
    let norm = RmsNorm::new(&tape, Tensor::filled([2], 1.0_f64), Tensor::filled([], 0.0));
    let input = tape.leaf(Tensor::new([3, 2], vec![1.0; 6]));
    let nodes_before = tape.len();

    norm.express(&tape, input);

    // Ten computed nodes plus the one count literal the mean records;
    // the total does not grow with batch or feature sizes.
    assert_eq!(tape.len(), nodes_before + 11);
}

#[test]
#[should_panic(expected = "disagree on features")]
fn express_rejects_mismatched_features() {
    let tape = Tape::new();
    let norm = RmsNorm::new(&tape, Tensor::filled([2], 1.0_f64), Tensor::filled([], 0.0));
    let input = tape.leaf(Tensor::new([2, 3], vec![1.0; 6]));
    norm.express(&tape, input);
}

#[test]
fn gradients_flow_through_the_root_mean_square() {
    // One sample with two features, `x = [2, 0]` and epsilon 2: the
    // mean square is 2, the root is `sqrt(2 + 2) = 2`, and the first
    // output is `n0 = x0 / sqrt(mean(x^2) + eps)`. Its exact gradient
    // is `1/root - x0^2 / (2 * root^3) = 1/4` on `x0` and `0` on `x1`
    // (the cross term carries a factor of `x1`).
    let tape = Tape::new();
    let norm = RmsNorm::new(&tape, Tensor::filled([2], 1.0_f64), Tensor::filled([], 2.0));
    let input = tape.leaf(Tensor::new([1, 2], [2.0, 0.0]));

    let output = norm.express(&tape, input);
    let target = output.narrow(1, 0, 1).sum();

    let (target, input) = (target.symbol(), input.symbol());
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(target);

    assert_eq!(gradients.of(input).to_vec(), &[0.25, 0.0]);

    // The scale sees the normalized value on the selected feature.
    let parameters: Vec<_> = norm.parameters().collect();
    assert_eq!(gradients.of(parameters[0]).to_vec(), &[1.0, 0.0]);
}
