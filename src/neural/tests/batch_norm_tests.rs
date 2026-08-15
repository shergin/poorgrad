use crate::{Network, Shape, Tensor};

use super::BatchNorm;

#[test]
fn new_allocates_scale_shift_and_epsilon() {
    let network = Network::new();
    let norm = BatchNorm::new(
        &network,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 1e-5),
    );
    // Two parameters and the epsilon constant, regardless of size.
    assert_eq!(network.len(), 3);
    assert_eq!(norm.parameters().count(), 2);
}

#[test]
#[should_panic(expected = "must be rank 1")]
fn new_rejects_non_vector_scale() {
    let network = Network::new();
    BatchNorm::new(
        &network,
        Tensor::filled([2, 2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 1e-5),
    );
}

#[test]
#[should_panic(expected = "shaped like the scale")]
fn new_rejects_mismatched_shift() {
    let network = Network::new();
    BatchNorm::new(
        &network,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([3], 0.0),
        Tensor::filled([], 1e-5),
    );
}

#[test]
#[should_panic(expected = "single value")]
fn new_rejects_multi_value_epsilon() {
    let network = Network::new();
    BatchNorm::new(
        &network,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([2], 1e-5),
    );
}

#[test]
fn express_standardizes_every_feature() {
    let network = Network::new();
    let norm = BatchNorm::new(
        &network,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 0.0),
    );
    // Feature columns `[1, 3]` and `[2, 6]`: means `[2, 4]`, biased
    // variances `[1, 4]`, so both standardize to `[-1, 1]`.
    let input = network.leaf(Tensor::new([2, 2], [1.0, 2.0, 3.0, 6.0]));

    let normalization = norm.express(&network, input);
    assert_eq!(normalization.output.shape(), Shape::new([2, 2]));
    assert_eq!(normalization.mean.shape(), Shape::new([2]));
    assert_eq!(normalization.variance.shape(), Shape::new([2]));

    let run = network.forward();
    assert_eq!(run.of(normalization.mean).to_vec(), &[2.0, 4.0]);
    assert_eq!(run.of(normalization.variance).to_vec(), &[1.0, 4.0]);
    assert_eq!(
        run.of(normalization.output).to_vec(),
        &[-1.0, -1.0, 1.0, 1.0]
    );
}

#[test]
fn express_applies_the_learned_affine() {
    let network = Network::new();
    let norm = BatchNorm::new(
        &network,
        Tensor::new([2], [2.0_f64, 3.0]),
        Tensor::new([2], [10.0, 20.0]),
        Tensor::filled([], 0.0),
    );
    let input = network.leaf(Tensor::new([2, 2], [1.0, 2.0, 3.0, 6.0]));

    let normalization = norm.express(&network, input);

    let run = network.forward();
    assert_eq!(
        run.of(normalization.output).to_vec(),
        &[8.0, 17.0, 12.0, 23.0]
    );
}

#[test]
fn epsilon_keeps_a_constant_feature_finite() {
    let network = Network::new();
    let norm = BatchNorm::new(
        &network,
        Tensor::filled([1], 1.0_f64),
        Tensor::filled([1], 0.0),
        Tensor::filled([], 4.0),
    );
    // A feature with no spread: zero variance would divide by zero
    // without the epsilon under the square root.
    let input = network.leaf(Tensor::new([2, 1], [5.0, 5.0]));

    let normalization = norm.express(&network, input);

    let run = network.forward();
    assert_eq!(run.of(normalization.output).to_vec(), &[0.0, 0.0]);
}

#[test]
fn express_records_tensor_granularity() {
    let network = Network::new();
    let norm = BatchNorm::new(
        &network,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 0.0),
    );
    let input = network.leaf(Tensor::new([3, 2], vec![1.0; 6]));
    let nodes_before = network.len();

    norm.express(&network, input);

    // Sixteen computed nodes plus the two count literals the means
    // record; the total does not grow with batch or feature sizes.
    assert_eq!(network.len(), nodes_before + 18);
}

#[test]
fn express_with_normalizes_by_the_fed_statistics() {
    let network = Network::new();
    let norm = BatchNorm::new(
        &network,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 0.0),
    );
    let input = network.input(Tensor::filled([1, 2], 0.0));
    let mean = network.input(Tensor::filled([2], 0.0));
    let variance = network.input(Tensor::filled([2], 1.0));

    let output = norm.express_with(&network, input, mean, variance);

    let run = network.forward_with([
        (input.symbol(), Tensor::new([1, 2], [3.0, 8.0])),
        (mean.symbol(), Tensor::new([2], [1.0, 4.0])),
        (variance.symbol(), Tensor::new([2], [4.0, 16.0])),
    ]);
    assert_eq!(run.of(output).to_vec(), &[1.0, 1.0]);

    // A later run feeds updated running estimates through the same
    // recorded expression.
    let run = network.forward_with([
        (input.symbol(), Tensor::new([1, 2], [3.0, 8.0])),
        (mean.symbol(), Tensor::new([2], [3.0, 0.0])),
        (variance.symbol(), Tensor::new([2], [1.0, 4.0])),
    ]);
    assert_eq!(run.of(output).to_vec(), &[0.0, 4.0]);
}

#[test]
#[should_panic(expected = "disagree on features")]
fn express_rejects_mismatched_features() {
    let network = Network::new();
    let norm = BatchNorm::new(
        &network,
        Tensor::filled([2], 1.0_f64),
        Tensor::filled([2], 0.0),
        Tensor::filled([], 0.0),
    );
    let input = network.leaf(Tensor::new([2, 3], vec![1.0; 6]));
    norm.express(&network, input);
}

#[test]
fn gradients_flow_through_the_batch_statistics() {
    // One feature over a batch of two, `x = [3, 1]`: the mean is 2, the
    // centered values are `[1, -1]`, the biased variance is 1, and with
    // epsilon 3 the deviation is 2. The first output is
    // `n0 = c / sqrt(c^2 + eps)` for `c = (x0 - x1) / 2`, whose exact
    // input gradient is `[3/16, -3/16]` — nonzero only because the
    // gradient also flows through the mean and variance.
    let network = Network::new();
    let norm = BatchNorm::new(
        &network,
        Tensor::filled([1], 1.0_f64),
        Tensor::filled([1], 0.0),
        Tensor::filled([], 3.0),
    );
    let input = network.leaf(Tensor::new([2, 1], [3.0, 1.0]));

    let normalization = norm.express(&network, input);
    let target = normalization.output.narrow(0, 0, 1).sum();

    let run = network.forward();
    let gradients = run.backward(target);

    let computed = gradients.of(input).to_vec();
    assert!((computed[0] - 0.1875).abs() < 1e-12);
    assert!((computed[1] + 0.1875).abs() < 1e-12);

    // The affine parameters receive the plain chain-rule shares: the
    // scale sees the normalized value and the shift sees the seed.
    let parameters: Vec<_> = norm.parameters().collect();
    let scale = network.resolve(parameters[0]);
    let shift = network.resolve(parameters[1]);
    assert_eq!(gradients.of(scale).to_vec(), &[0.5]);
    assert_eq!(gradients.of(shift).to_vec(), &[1.0]);
}
