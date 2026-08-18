use crate::{Shape, Tape, Tensor};

use super::LayerNorm;

#[test]
fn new_allocates_scale_shift_and_epsilon() {
    let tape = Tape::new();
    let norm = LayerNorm::new(
        &tape,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 1e-5),
    );
    // Two parameters and the epsilon constant, regardless of size.
    assert_eq!(tape.len(), 3);
    assert_eq!(norm.parameters().count(), 2);
}

#[test]
#[should_panic(expected = "must be rank 1")]
fn new_rejects_non_vector_scale() {
    let tape = Tape::new();
    LayerNorm::new(
        &tape,
        Tensor::filled([2, 2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 1e-5),
    );
}

#[test]
#[should_panic(expected = "single value")]
fn new_rejects_multi_value_epsilon() {
    let tape = Tape::new();
    LayerNorm::new(
        &tape,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([2], 1e-5),
    );
}

#[test]
fn express_standardizes_every_sample() {
    let tape = Tape::new();
    let norm = LayerNorm::new(
        &tape,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 0.0),
    );
    // Sample rows `[1, 3]` and `[2, 6]`: means `[2, 4]`, biased
    // variances `[1, 4]`, so both standardize to `[-1, 1]`.
    let input = tape.leaf(Tensor::new([2, 2], [1.0, 3.0, 2.0, 6.0]));

    let output = norm.express(&tape, input);
    assert_eq!(output.shape(), Shape::new([2, 2]));

    let output = output.symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(output).to_vec(), &[-1.0, 1.0, -1.0, 1.0]);
}

#[test]
fn express_applies_the_learned_affine() {
    let tape = Tape::new();
    let norm = LayerNorm::new(
        &tape,
        Tensor::new([2], [2.0_f64, 3.0]),
        Tensor::new([2], [10.0, 20.0]),
        Tensor::filled([], 0.0),
    );
    let input = tape.leaf(Tensor::new([2, 2], [1.0, 3.0, 2.0, 6.0]));

    let output = norm.express(&tape, input).symbol();

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(output).to_vec(), &[8.0, 23.0, 8.0, 23.0]);
}

#[test]
fn samples_normalize_independently() {
    // Unlike batch normalization, one sample's output is a function of
    // that sample alone: the same row yields the same output whatever
    // shares the batch with it.
    let tape = Tape::new();
    let norm = LayerNorm::new(
        &tape,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 0.0),
    );
    let lone = tape.leaf(Tensor::new([1, 2], [1.0, 3.0]));
    let paired = tape.leaf(Tensor::new([2, 2], [1.0, 3.0, -100.0, 900.0]));

    let lone_output = norm.express(&tape, lone).symbol();
    let paired_output = norm.express(&tape, paired).symbol();

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(
        run.of(lone_output).to_vec(),
        run.of(paired_output).to_vec()[..2]
    );
}

#[test]
fn express_records_tensor_granularity() {
    let tape = Tape::new();
    let norm = LayerNorm::new(
        &tape,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 0.0),
    );
    let input = tape.leaf(Tensor::new([3, 2], vec![1.0; 6]));
    let nodes_before = tape.len();

    norm.express(&tape, input);

    // Sixteen computed nodes plus the two count literals the means
    // record; the total does not grow with batch or feature sizes.
    assert_eq!(tape.len(), nodes_before + 18);
}

#[test]
#[should_panic(expected = "disagree on features")]
fn express_rejects_mismatched_features() {
    let tape = Tape::new();
    let norm = LayerNorm::new(
        &tape,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 0.0),
    );
    let input = tape.leaf(Tensor::new([2, 3], vec![1.0; 6]));
    norm.express(&tape, input);
}

#[test]
fn gradients_flow_through_the_sample_statistics() {
    // One sample with two features, `x = [3, 1]`: the mean is 2, the
    // centered values are `[1, -1]`, the biased variance is 1, and with
    // epsilon 3 the deviation is 2. The first output is
    // `n0 = c / sqrt(c^2 + eps)` for `c = (x0 - x1) / 2`, whose exact
    // input gradient is `[3/16, -3/16]` — the transpose of the
    // batch-norm case, since the statistics run along the other axis.
    let tape = Tape::new();
    let norm = LayerNorm::new(
        &tape,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 3.0),
    );
    let input = tape.leaf(Tensor::new([1, 2], [3.0, 1.0]));

    let output = norm.express(&tape, input);
    let target = output.narrow(1, 0, 1).sum();

    let (target, input) = (target.symbol(), input.symbol());
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(target);

    let computed = gradients.of(input).to_vec();
    assert!((computed[0] - 0.1875).abs() < 1e-12);
    assert!((computed[1] + 0.1875).abs() < 1e-12);

    // The affine parameters receive the plain chain-rule shares on the
    // selected feature alone: the scale sees the normalized value and
    // the shift sees the seed.
    let parameters: Vec<_> = norm.parameters().collect();
    assert_eq!(gradients.of(parameters[0]).to_vec(), &[0.5, 0.0]);
    assert_eq!(gradients.of(parameters[1]).to_vec(), &[1.0, 0.0]);
}
