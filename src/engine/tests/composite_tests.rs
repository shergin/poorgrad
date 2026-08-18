use crate::{Shape, Tape, Tensor, concat, stack};

#[test]
fn abs_composes_from_maximum() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([3], [-2.0_f64, 0.0, 3.0]));
    let magnitude = x.abs();
    let loss = magnitude.sum();
    let (x, magnitude, loss) = (x.symbol(), magnitude.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(magnitude).to_vec(), &[2.0, 0.0, 3.0]);

    let gradients = run.backward(loss);
    assert_eq!(gradients.of(x).to_vec(), &[-1.0, 1.0, 1.0]);
}

#[test]
fn softmax_matches_the_probabilities() {
    let tape = Tape::new();
    let logits = tape.leaf(Tensor::new([1, 2], [0.0_f64, 3.0_f64.ln()]));
    let probabilities = logits.softmax(1).symbol();
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    let expected = [0.25, 0.75];
    for (computed, expected) in run.of(probabilities).to_vec().into_iter().zip(expected) {
        assert!((computed - expected).abs() < 1e-12);
    }
}

#[test]
fn softmax_inherits_stability_from_the_fused_core() {
    let tape = Tape::new();
    // Naive softmax overflows at `exp(1000)`; through the fused
    // log-softmax the probabilities stay exact.
    let logits = tape.leaf(Tensor::new([1, 2], [1000.0_f64, 1000.0]));
    let probabilities = logits.softmax(1).symbol();
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    for probability in run.of(probabilities).to_vec() {
        assert!((probability - 0.5).abs() < 1e-12);
    }
}

#[test]
fn logsumexp_reduces_like_a_smooth_maximum() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2, 2], [0.0_f64, 3.0_f64.ln(), 1000.0, 1000.0]));
    let reduced = x.logsumexp(1);
    assert_eq!(reduced.shape(), Shape::new([2]));
    let reduced = reduced.symbol();
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    let values = run.of(reduced).to_vec();
    assert!((values[0] - 4.0_f64.ln()).abs() < 1e-12);
    // The second row would overflow a naive `ln(sum(exp(x)))`.
    assert!((values[1] - (1000.0 + 2.0_f64.ln())).abs() < 1e-12);
}

#[test]
fn logsumexp_stays_finite_for_finite_extreme_logits() {
    // The finite difference overflows the representable range; the
    // mathematical answer is approximately the maximum and must stay
    // finite in either lane order.
    for logits in [[-1.0e308_f64, 1.0e308], [1.0e308, -1.0e308]] {
        let tape = Tape::new();
        let x = tape.leaf(Tensor::new([2], logits));
        let reduced = x.logsumexp(0).symbol();
        let network = tape.into_network();
        let run = network.forward(&network.parameters(), []);
        let value = run.of(reduced).to_vec()[0];
        assert!(value.is_finite());
        assert!((value - 1.0e308).abs() < 1.0e293);
    }
}

#[test]
fn mean_along_divides_by_the_axis_extent() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let mean = x.mean_along(0);
    assert_eq!(mean.shape(), Shape::new([3]));
    let loss = mean.sum();
    let (x, mean, loss) = (x.symbol(), mean.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(mean).to_vec(), &[2.5, 3.5, 4.5]);

    // Each sample contributes `1 / extent` to the mean's gradient.
    let gradients = run.backward(loss);
    assert_eq!(gradients.of(x).to_vec(), &[0.5; 6]);
}

#[test]
#[should_panic(expected = "out of rank")]
fn mean_along_rejects_an_axis_out_of_rank() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2], [1.0_f64, 2.0]));
    x.mean_along(1);
}

#[test]
fn broadcast_to_prepends_leading_axes() {
    let tape = Tape::new();
    let row = tape.leaf(Tensor::new([3], [1.0_f64, 2.0, 3.0]));
    let grid = row.broadcast_to([2, 3]);
    assert_eq!(grid.shape(), Shape::new([2, 3]));
    let loss = grid.sum();
    let (row, grid, loss) = (row.symbol(), grid.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(grid).to_vec(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

    // Each source element feeds both rows, so its gradient is the count of
    // rows it was repeated across.
    let gradients = run.backward(loss);
    assert_eq!(gradients.of(row).to_vec(), &[2.0, 2.0, 2.0]);
}

#[test]
fn broadcast_to_expands_interior_unit_axes() {
    let tape = Tape::new();
    let column = tape.leaf(Tensor::new([2, 1, 2], [1.0_f64, 2.0, 3.0, 4.0]));
    let grid = column.broadcast_to([2, 3, 2]);
    assert_eq!(grid.shape(), Shape::new([2, 3, 2]));
    let loss = grid.sum();
    let (column, grid, loss) = (column.symbol(), grid.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(
        run.of(grid).to_vec(),
        &[1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0, 3.0, 4.0]
    );

    // The extent-one axis is repeated three times.
    let gradients = run.backward(loss);
    assert_eq!(gradients.of(column).to_vec(), &[3.0, 3.0, 3.0, 3.0]);
}

#[test]
fn broadcast_to_expands_several_axes() {
    let tape = Tape::new();
    let row = tape.leaf(Tensor::new([1, 3], [1.0_f64, 2.0, 3.0]));
    let block = row.broadcast_to([2, 2, 3]);
    assert_eq!(block.shape(), Shape::new([2, 2, 3]));
    let loss = block.sum();
    let (row, block, loss) = (row.symbol(), block.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(block).to_vec(), [1.0, 2.0, 3.0].repeat(4));

    // The source feeds all four repeated rows.
    let gradients = run.backward(loss);
    assert_eq!(gradients.of(row).to_vec(), &[4.0, 4.0, 4.0]);
}

#[test]
fn broadcast_to_is_identity_on_an_equal_shape() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]));
    let same = x.broadcast_to([2, 2]);
    assert_eq!(same.shape(), Shape::new([2, 2]));
    let same = same.symbol();
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(same).to_vec(), &[1.0, 2.0, 3.0, 4.0]);
}

#[test]
#[should_panic(expected = "cannot align")]
fn broadcast_to_rejects_an_incompatible_axis() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([3], [1.0_f64, 2.0, 3.0]));
    x.broadcast_to([2, 4]);
}

#[test]
fn broadcast_pair_lifts_both_operands_to_the_common_shape() {
    let tape = Tape::new();
    // Outer sum of a column and a row: [2, 1] against [1, 3] gives [2, 3].
    let column = tape.leaf(Tensor::new([2, 1], [1.0_f64, 2.0]));
    let row = tape.leaf(Tensor::new([1, 3], [10.0_f64, 20.0, 30.0]));
    let (left, right) = column.broadcast_pair(row);
    assert_eq!(left.shape(), Shape::new([2, 3]));
    assert_eq!(right.shape(), Shape::new([2, 3]));
    let sum = left + right;
    let loss = sum.sum();
    let (column, row, sum, loss) = (column.symbol(), row.symbol(), sum.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(sum).to_vec(), &[11.0, 21.0, 31.0, 12.0, 22.0, 32.0]);

    let gradients = run.backward(loss);
    // The column repeats across three columns, the row across two rows.
    assert_eq!(gradients.of(column).to_vec(), &[3.0, 3.0]);
    assert_eq!(gradients.of(row).to_vec(), &[2.0, 2.0, 2.0]);
}

#[test]
fn logsumexp_gradient_is_the_softmax() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([1, 2], [0.0_f64, 3.0_f64.ln()]));
    let loss = x.logsumexp(1).sum();
    let (x, loss) = (x.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    let gradients = run.backward(loss);
    let expected = [0.25, 0.75];
    for (computed, expected) in gradients.of(x).to_vec().into_iter().zip(expected) {
        assert!((computed - expected).abs() < 1e-12);
    }
}

#[test]
fn concat_joins_values_along_the_leading_axis() {
    let tape = Tape::new();
    let top = tape.leaf(Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]));
    let bottom = tape.leaf(Tensor::new([1, 2], [5.0_f64, 6.0]));
    let joined = concat(&[top, bottom], 0);
    assert_eq!(joined.shape(), Shape::new([3, 2]));

    let weights = tape.leaf(Tensor::new([3, 2], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let loss = (joined * weights).sum();
    let (top, bottom, joined, loss) = (
        top.symbol(),
        bottom.symbol(),
        joined.symbol(),
        loss.symbol(),
    );
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(joined).to_vec(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    // Each operand's gradient is the weight window it occupies.
    let gradients = run.backward(loss);
    assert_eq!(gradients.of(top).to_vec(), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(gradients.of(bottom).to_vec(), &[5.0, 6.0]);
}

#[test]
fn concat_joins_values_along_an_interior_axis() {
    let tape = Tape::new();
    let left = tape.leaf(Tensor::new([2, 1], [1.0_f64, 4.0]));
    let right = tape.leaf(Tensor::new([2, 2], [2.0_f64, 3.0, 5.0, 6.0]));
    let joined = concat(&[left, right], 1);
    assert_eq!(joined.shape(), Shape::new([2, 3]));
    let joined = joined.symbol();
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(joined).to_vec(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn concat_of_a_single_value_is_the_value_itself() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2], [1.0_f64, 2.0]));
    let joined = concat(&[x], 0);
    assert_eq!(joined.shape(), Shape::new([2]));
    let joined = joined.symbol();
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(joined).to_vec(), &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "out of rank")]
fn concat_rejects_an_axis_out_of_rank() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2], [1.0_f64, 2.0]));
    concat(&[x, x], 1);
}

#[test]
#[should_panic(expected = "equal shapes off the axis")]
fn concat_rejects_mismatched_shapes_off_the_axis() {
    let tape = Tape::new();
    let x = tape.leaf(Tensor::new([2, 2], vec![1.0_f64; 4]));
    let y = tape.leaf(Tensor::new([2, 3], vec![1.0_f64; 6]));
    concat(&[x, y], 0);
}

#[test]
fn stack_lifts_values_onto_a_new_axis() {
    let tape = Tape::new();
    let first = tape.leaf(Tensor::new([3], [1.0_f64, 2.0, 3.0]));
    let second = tape.leaf(Tensor::new([3], [4.0_f64, 5.0, 6.0]));

    let rows = stack(&[first, second], 0);
    assert_eq!(rows.shape(), Shape::new([2, 3]));
    let columns = stack(&[first, second], 1);
    assert_eq!(columns.shape(), Shape::new([3, 2]));
    let loss = rows.sum();
    let (first, second, rows, columns, loss) = (
        first.symbol(),
        second.symbol(),
        rows.symbol(),
        columns.symbol(),
        loss.symbol(),
    );
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(rows).to_vec(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(run.of(columns).to_vec(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let gradients = run.backward(loss);
    assert_eq!(gradients.of(first).to_vec(), &[1.0, 1.0, 1.0]);
    assert_eq!(gradients.of(second).to_vec(), &[1.0, 1.0, 1.0]);
}

#[test]
fn masked_softmax_composes_with_a_broadcast_mask() {
    // The transformer rung's "masked axis-aware softmax" gap closes by
    // composition: an additive mask spread over the scores, then the
    // existing axis softmax. No dedicated operation is required.
    let tape = Tape::new();
    let scores = tape.leaf(Tensor::new([2, 3], [1.0_f64, 1.0, 9.0, 2.0, 2.0, 9.0]));
    let mask = tape.leaf(Tensor::new([3], [0.0_f64, 0.0, f64::NEG_INFINITY]));
    let probabilities = (scores + mask.broadcast_to([2, 3])).softmax(1);
    let loss = probabilities.narrow(1, 0, 1).sum();
    let (scores, probabilities, loss) = (scores.symbol(), probabilities.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    let computed = run.of(probabilities).to_vec();
    let expected = [0.5, 0.5, 0.0, 0.5, 0.5, 0.0];
    for (computed, expected) in computed.into_iter().zip(expected) {
        assert!((computed - expected).abs() < 1e-12);
    }

    // Row-wise softmax gradient of the first probability: p0 * (1 - p0)
    // for itself, -p0 * p1 for its live neighbor, zero for the masked
    // lane.
    let gradients = run.backward(loss);
    let expected = [0.25, -0.25, 0.0, 0.25, -0.25, 0.0];
    for (computed, expected) in gradients.of(scores).to_vec().into_iter().zip(expected) {
        assert!((computed - expected).abs() < 1e-12);
    }
}

#[test]
fn attention_heads_compose_with_a_loop_and_concat() {
    // The rung's head story without batched matmul: each head is a
    // rank-2 attention recorded in a loop, and `concat` joins the head
    // outputs along the feature axis.
    let tape = Tape::new();
    let causal = tape.leaf(Tensor::new([2, 2], [0.0_f64, f64::NEG_INFINITY, 0.0, 0.0]));
    let mut heads = Vec::new();
    let mut leaves = Vec::new();
    for seed in 0..2 {
        let shift = seed as f64;
        let query = tape.leaf(Tensor::new([2, 2], [shift, 1.0, 0.0, 1.0]));
        let key = tape.leaf(Tensor::new([2, 2], [1.0_f64, 0.0, shift, 1.0]));
        let value = tape.leaf(Tensor::new([2, 2], [1.0 + shift, 2.0, 3.0, 4.0 + shift]));
        let weights = (query.matmul(key.transpose()) + causal).softmax(1);
        heads.push(weights.matmul(value));
        leaves.push((query.symbol(), key.symbol(), value.symbol()));
    }
    let output = concat(&heads, 1);
    assert_eq!(output.shape(), Shape::new([2, 4]));
    let loss = output.sum();
    let (output, loss) = (output.symbol(), loss.symbol());
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    // The causal mask pins the first token to its own value row in every
    // head.
    let computed = run.of(output).to_vec();
    assert!((computed[0] - 1.0).abs() < 1e-12);
    assert!((computed[1] - 2.0).abs() < 1e-12);
    assert!((computed[2] - 2.0).abs() < 1e-12);
    assert!((computed[3] - 2.0).abs() < 1e-12);

    // Every head's projections receive gradient through the concat.
    let gradients = run.backward(loss);
    for (query, key, value) in leaves {
        assert!(gradients.of(query).to_vec().iter().any(|&g| g != 0.0));
        assert_eq!(gradients.of(value).to_vec().len(), 4);
        let _ = key;
    }
}
