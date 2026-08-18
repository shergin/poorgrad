use crate::{Shape, Tape, Tensor};

use super::cross_entropy;

#[test]
fn uniform_logits_cost_the_log_of_the_class_count() {
    let tape = Tape::new();
    let logits = tape.leaf(Tensor::filled([4, 3], 0.0_f64));
    let targets = tape.input(Tensor::selection(vec![0usize, 1, 2, 0], 3, 1.0));

    let loss = cross_entropy(logits, targets);
    assert_eq!(loss.shape(), Shape::scalar());

    let loss = loss.symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let cost = run.of(loss).to_vec()[0];
    assert!((cost - 3.0_f64.ln()).abs() < 1e-12);
}

#[test]
fn confident_correct_logits_cost_nothing() {
    let tape = Tape::new();
    // The extreme margin would overflow a naive softmax; the fused
    // log-softmax keeps the loss an exact zero.
    let logits = tape.leaf(Tensor::new([1, 2], [1000.0_f64, -1000.0]));
    let targets = tape.input(Tensor::selection(vec![0usize], 2, 1.0));

    let loss = cross_entropy(logits, targets).symbol();

    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let cost = run.of(loss).to_vec()[0];
    assert!(cost.is_finite());
    assert!(cost.abs() < 1e-12);
}

#[test]
fn gradient_is_probabilities_minus_targets_over_the_batch() {
    let tape = Tape::new();
    // Row softmaxes are `[0.25, 0.75]` and `[0.5, 0.5]`; with targets `0`
    // and `1` the mean-loss gradient is `(softmax - onehot) / batch`.
    let logits = tape.parameter(Tensor::new([2, 2], [0.0_f64, 3.0_f64.ln(), 0.0, 0.0]));
    let targets = tape.input(Tensor::selection(vec![0usize, 1], 2, 1.0));

    let loss = cross_entropy(logits, targets);

    let (loss, logits) = (loss.symbol(), logits.symbol());
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(loss);
    let expected = [-0.375, 0.375, 0.25, -0.25];
    for (computed, expected) in gradients.of(logits).to_vec().into_iter().zip(expected) {
        assert!((computed - expected).abs() < 1e-12);
    }
}

#[test]
fn served_batches_vary_per_run() {
    let tape = Tape::new();
    let logits = tape.input(Tensor::filled([2, 3], 0.0_f64));
    let targets = tape.input(Tensor::selection(vec![0usize, 1], 3, 1.0));
    let logits_symbol = logits.symbol();
    let targets_symbol = targets.symbol();

    let loss = cross_entropy(logits, targets).symbol();
    let network = tape.into_network();

    // A batch that puts all its mass on the labeled classes drives the loss
    // toward zero, while the recorded graph stays fixed.
    let confident = Tensor::new([2, 3], [50.0_f64, 0.0, 0.0, 0.0, 50.0, 0.0]);
    let labels = Tensor::selection(vec![0usize, 1], 3, 1.0);
    let run = network.forward(
        &network.parameters(),
        [(logits_symbol, confident), (targets_symbol, labels)],
    );
    let cost = run.of(loss).to_vec()[0];
    assert!(cost.abs() < 1e-12);
}

#[test]
#[should_panic(expected = "must be rank 2")]
fn rejects_non_matrix_logits() {
    let tape = Tape::new();
    let logits = tape.leaf(Tensor::filled([3], 0.0_f64));
    let targets = tape.input(Tensor::selection(vec![0usize], 3, 1.0));
    cross_entropy(logits, targets);
}

#[test]
#[should_panic(expected = "must be shaped like the logits")]
fn rejects_mismatched_targets() {
    let tape = Tape::new();
    let logits = tape.leaf(Tensor::filled([2, 3], 0.0_f64));
    let targets = tape.input(Tensor::selection(vec![0usize], 3, 1.0));
    cross_entropy(logits, targets);
}

#[test]
fn extreme_finite_logits_keep_the_loss_and_gradients_finite() {
    // Finite logits whose difference overflows the representable range,
    // with the correct class overwhelmingly likely: the loss is exactly
    // zero and no gradient lane may turn NaN — in either class order,
    // for either selected class.
    for (row, class) in [([-1.0e308_f64, 1.0e308], 1_usize), ([1.0e308, -1.0e308], 0)] {
        let tape = Tape::new();
        let logits = tape.parameter(Tensor::new([1, 2], row));
        let targets = tape.input(Tensor::selection(vec![class], 2, 1.0));

        let loss = cross_entropy(logits, targets);
        let (loss, logits) = (loss.symbol(), logits.symbol());
        let network = tape.into_network();
        let run = network.forward(&network.parameters(), []);
        assert_eq!(run.of(loss).to_vec()[0], 0.0);

        let gradients = run.backward(loss);
        for gradient in gradients.of(logits).to_vec() {
            assert!(gradient.is_finite());
        }
    }
}

#[test]
fn extreme_finite_logits_keep_the_loss_finite_f32() {
    let tape = Tape::new();
    let logits = tape.leaf(Tensor::new([1, 2], [-3.0e38_f32, 3.0e38]));
    let targets = tape.input(Tensor::selection(vec![1_usize], 2, 1.0));

    let loss = cross_entropy(logits, targets).symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(loss).to_vec()[0], 0.0);
}

#[test]
fn zero_target_lanes_contribute_exact_zero() {
    let tape = Tape::new();
    // A dense soft target with an explicit zero lane against the logit
    // whose log-probability underflows to -inf: the zero lane must
    // contribute zero, never `0 * -inf = NaN`.
    let logits = tape.leaf(Tensor::new([1, 2], [-1.0e308_f64, 1.0e308]));
    let targets = tape.input(Tensor::new([1, 2], [0.0_f64, 1.0]));

    let loss = cross_entropy(logits, targets).symbol();
    let network = tape.into_network();
    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(loss).to_vec()[0], 0.0);
}
