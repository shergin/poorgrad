use crate::{Network, Shape, Tensor};

#[test]
fn abs_composes_from_maximum() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([3], [-2.0_f64, 0.0, 3.0]));
    let magnitude = x.abs();
    let loss = magnitude.sum();

    let evaluation = network.forward();
    assert_eq!(evaluation.of(magnitude).to_vec(), &[2.0, 0.0, 3.0]);

    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(x).to_vec(), &[-1.0, 1.0, 1.0]);
}

#[test]
fn softmax_matches_the_probabilities() {
    let network = Network::new();
    let logits = network.leaf(Tensor::new([1, 2], [0.0_f64, 3.0_f64.ln()]));
    let probabilities = logits.softmax(1);

    let evaluation = network.forward();
    let expected = [0.25, 0.75];
    for (computed, expected) in evaluation
        .of(probabilities)
        .to_vec()
        .into_iter()
        .zip(expected)
    {
        assert!((computed - expected).abs() < 1e-12);
    }
}

#[test]
fn softmax_inherits_stability_from_the_fused_core() {
    let network = Network::new();
    // Naive softmax overflows at `exp(1000)`; through the fused
    // log-softmax the probabilities stay exact.
    let logits = network.leaf(Tensor::new([1, 2], [1000.0_f64, 1000.0]));
    let probabilities = logits.softmax(1);

    let evaluation = network.forward();
    for probability in evaluation.of(probabilities).to_vec() {
        assert!((probability - 0.5).abs() < 1e-12);
    }
}

#[test]
fn logsumexp_reduces_like_a_smooth_maximum() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2, 2], [0.0_f64, 3.0_f64.ln(), 1000.0, 1000.0]));
    let reduced = x.logsumexp(1);
    assert_eq!(reduced.shape(), Shape::new([2]));

    let evaluation = network.forward();
    let values = evaluation.of(reduced).to_vec();
    assert!((values[0] - 4.0_f64.ln()).abs() < 1e-12);
    // The second row would overflow a naive `ln(sum(exp(x)))`.
    assert!((values[1] - (1000.0 + 2.0_f64.ln())).abs() < 1e-12);
}

#[test]
fn mean_along_divides_by_the_axis_extent() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let mean = x.mean_along(0);
    assert_eq!(mean.shape(), Shape::new([3]));
    let loss = mean.sum();

    let evaluation = network.forward();
    assert_eq!(evaluation.of(mean).to_vec(), &[2.5, 3.5, 4.5]);

    // Each sample contributes `1 / extent` to the mean's gradient.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(x).to_vec(), &[0.5; 6]);
}

#[test]
#[should_panic(expected = "out of rank")]
fn mean_along_rejects_an_axis_out_of_rank() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2], [1.0_f64, 2.0]));
    x.mean_along(1);
}

#[test]
fn logsumexp_gradient_is_the_softmax() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([1, 2], [0.0_f64, 3.0_f64.ln()]));
    let loss = x.logsumexp(1).sum();

    let evaluation = network.forward();
    let gradients = evaluation.backward(loss);
    let expected = [0.25, 0.75];
    for (computed, expected) in gradients.of(x).to_vec().into_iter().zip(expected) {
        assert!((computed - expected).abs() < 1e-12);
    }
}
