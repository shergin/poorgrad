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
fn broadcast_to_prepends_leading_axes() {
    let network = Network::new();
    let row = network.leaf(Tensor::new([3], [1.0_f64, 2.0, 3.0]));
    let grid = row.broadcast_to([2, 3]);
    assert_eq!(grid.shape(), Shape::new([2, 3]));
    let loss = grid.sum();

    let evaluation = network.forward();
    assert_eq!(
        evaluation.of(grid).to_vec(),
        &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    );

    // Each source element feeds both rows, so its gradient is the count of
    // rows it was repeated across.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(row).to_vec(), &[2.0, 2.0, 2.0]);
}

#[test]
fn broadcast_to_expands_interior_unit_axes() {
    let network = Network::new();
    let column = network.leaf(Tensor::new([2, 1, 2], [1.0_f64, 2.0, 3.0, 4.0]));
    let grid = column.broadcast_to([2, 3, 2]);
    assert_eq!(grid.shape(), Shape::new([2, 3, 2]));
    let loss = grid.sum();

    let evaluation = network.forward();
    assert_eq!(
        evaluation.of(grid).to_vec(),
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    );

    // The extent-one axis is repeated three times.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(column).to_vec(), &[3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn broadcast_to_expands_several_axes() {
    let network = Network::new();
    let row = network.leaf(Tensor::new([1, 3], [1.0_f64, 2.0, 3.0]));
    let block = row.broadcast_to([2, 2, 3]);
    assert_eq!(block.shape(), Shape::new([2, 2, 3]));
    let loss = block.sum();

    let evaluation = network.forward();
    assert_eq!(evaluation.of(block).to_vec(), [1.0, 2.0, 3.0].repeat(4));

    // The source feeds all four repeated rows.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(row).to_vec(), &[4.0, 4.0, 4.0]);
}

#[test]
fn broadcast_to_is_identity_on_an_equal_shape() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]));
    let same = x.broadcast_to([2, 2]);
    assert_eq!(same.shape(), Shape::new([2, 2]));

    let evaluation = network.forward();
    assert_eq!(evaluation.of(same).to_vec(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[should_panic(expected = "cannot align")]
fn broadcast_to_rejects_an_incompatible_axis() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([3], [1.0_f64, 2.0, 3.0]));
    x.broadcast_to([2, 4]);
}

#[test]
fn broadcast_pair_lifts_both_operands_to_the_common_shape() {
    let network = Network::new();
    // Outer sum of a column and a row: [2, 1] against [1, 3] gives [2, 3].
    let column = network.leaf(Tensor::new([2, 1], [1.0_f64, 2.0]));
    let row = network.leaf(Tensor::new([1, 3], [10.0_f64, 20.0, 30.0]));
    let (left, right) = column.broadcast_pair(row);
    assert_eq!(left.shape(), Shape::new([2, 3]));
    assert_eq!(right.shape(), Shape::new([2, 3]));
    let sum = left + right;
    let loss = sum.sum();

    let evaluation = network.forward();
    assert_eq!(
        evaluation.of(sum).to_vec(),
        &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]
    );

    let gradients = evaluation.backward(loss);
    // The column repeats across three columns, the row across two rows.
    assert_eq!(gradients.of(column).to_vec(), &[3.0, 3.0]);
    assert_eq!(gradients.of(row).to_vec(), &[2.0, 2.0, 2.0]);
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
