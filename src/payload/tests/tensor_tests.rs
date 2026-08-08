use crate::{Differentiable, Elementary, Network, Shape, Tensorial};

use super::Tensor;

#[test]
fn new_builds_from_shape_and_elements() {
    let tensor = Tensor::new([2, 3], vec![1.0_f64; 6]);
    assert_eq!(tensor.shape(), Shape::new([2, 3]));
    assert_eq!(tensor.to_vec().len(), 6);
}

#[test]
#[should_panic(expected = "shape does not match")]
fn new_rejects_mismatched_volume() {
    Tensor::new([2, 3], vec![1.0_f64; 5]);
}

#[test]
#[should_panic(expected = "at least one element")]
fn new_rejects_empty_tensors() {
    Tensor::new([2, 0], Vec::<f64>::new());
}

#[test]
#[should_panic(expected = "at least one element")]
fn filled_rejects_empty_tensors() {
    Tensor::filled([0], 1.0_f64);
}

#[test]
fn clone_shares_storage() {
    let tensor = Tensor::new([2], [1.0_f64, 2.0]);
    let clone = tensor.clone();
    assert!(tensor.as_slice().unwrap().as_ptr() == clone.as_slice().unwrap().as_ptr());
}

#[test]
fn arithmetic_applies_elementwise() {
    let left = Tensor::new([2], [1.0_f64, 2.0]);
    let right = Tensor::new([2], [10.0, 20.0]);

    assert_eq!((left.clone() + right.clone()).to_vec(), &[11.0, 22.0]);
    assert_eq!((right.clone() - left.clone()).to_vec(), &[9.0, 18.0]);
    assert_eq!((left.clone() * right.clone()).to_vec(), &[10.0, 40.0]);
    assert_eq!((right.clone() / left.clone()).to_vec(), &[10.0, 10.0]);
    assert_eq!((-left).to_vec(), &[-1.0, -2.0]);
}

#[test]
#[should_panic(expected = "different shapes")]
fn arithmetic_rejects_shape_mismatch() {
    let _ = Tensor::new([2], [1.0_f64, 2.0]) + Tensor::new([3], [1.0, 2.0, 3.0]);
}

#[test]
fn likes_preserve_shape() {
    let tensor = Tensor::new([2, 2], vec![7.0_f64; 4]);
    let zero = tensor.zero_like();
    let one = tensor.one_like();
    assert_eq!(zero.shape(), Shape::new([2, 2]));
    assert_eq!(zero.to_vec(), &[0.0; 4]);
    assert_eq!(one.to_vec(), &[1.0; 4]);
}

#[test]
fn counted_spreads_the_count_across_the_shape() {
    let counted = Tensor::<f64>::counted(Shape::new([2, 3]), 6);
    assert_eq!(counted.shape(), Shape::new([2, 3]));
    assert_eq!(counted.to_vec(), &[6.0; 6]);
}

#[test]
#[should_panic(expected = "at least one element")]
fn counted_rejects_empty_shapes() {
    Tensor::<f64>::counted(Shape::new([0]), 1);
}

#[test]
#[should_panic(expected = "volume overflows")]
fn counted_rejects_overflowing_shapes() {
    Tensor::<f64>::counted(Shape::new([usize::MAX, 2]), 1);
}

#[test]
#[should_panic(expected = "volume overflows")]
fn selection_rejects_overflowing_shapes() {
    Tensor::selection(vec![0_usize, 0], usize::MAX, 1.0_f64);
}

#[test]
fn unfold_single_window_accepts_any_step() {
    // The contract permits any positive step; with one window the start
    // stride is never applied, so even a step whose stride product would
    // overflow must behave identically in debug and release builds.
    let tensor = Tensor::new([1, 2], [1.0_f64, 2.0]);
    let windows = tensor.unfold(0, 1, usize::MAX, 1);
    assert_eq!(windows.shape(), Shape::new([1, 1, 2]));
    assert_eq!(windows.to_vec(), &[1.0, 2.0]);
}

#[test]
fn unfold_slides_overlapping_windows() {
    let tensor = Tensor::new([8], (1..=8).map(|v| v as f64).collect::<Vec<_>>());
    let windows = tensor.unfold(0, 3, 2, 1);
    assert_eq!(windows.shape(), Shape::new([3, 3]));
    // Windows start every 2 elements and overlap by one; the view is
    // strided, not contiguous.
    assert_eq!(
        windows.to_vec(),
        &[1.0, 2.0, 3.0, 3.0, 4.0, 5.0, 5.0, 6.0, 7.0]
    );
    assert!(windows.as_slice().is_none());
}

#[test]
fn unfold_dilation_spaces_the_window_elements() {
    let tensor = Tensor::new([8], (1..=8).map(|v| v as f64).collect::<Vec<_>>());
    let windows = tensor.unfold(0, 2, 1, 3);
    // Span `3 * (2 - 1) + 1 = 4`, so `(8 - 4) / 1 + 1 = 5` windows of
    // elements three apart.
    assert_eq!(windows.shape(), Shape::new([5, 2]));
    assert_eq!(
        windows.to_vec(),
        &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 4.0, 7.0, 5.0, 8.0]
    );
}

#[test]
fn unfold_preserves_surrounding_axes() {
    let tensor = Tensor::new([2, 4], (1..=8).map(|v| v as f64).collect::<Vec<_>>());
    let windows = tensor.unfold(1, 2, 2, 1);
    assert_eq!(windows.shape(), Shape::new([2, 2, 2]));
    assert_eq!(windows.to_vec(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
}

#[test]
fn unfold_keeps_constants_constant() {
    let windows = Tensor::filled([5], 7.0_f64).unfold(0, 2, 1, 1);
    assert_eq!(windows.shape(), Shape::new([4, 2]));
    assert_eq!(windows.to_vec(), &[7.0; 8]);
}

#[test]
#[should_panic(expected = "exceeds axis 0 extent")]
fn unfold_rejects_an_oversized_window() {
    Tensor::new([4], vec![1.0_f64; 4]).unfold(0, 3, 2, 2);
}

#[test]
fn fold_sums_the_window_coverage() {
    // Folding all-ones windows counts how many windows read each
    // position; position 7 is beyond the last window and folds to zero.
    let windows = Tensor::filled([3, 3], 1.0_f64);
    let folded = windows.fold(0, 3, 2, 1, 8);
    assert_eq!(folded.shape(), Shape::new([8]));
    assert_eq!(folded.to_vec(), &[1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 0.0]);
}

#[test]
fn fold_is_the_adjoint_of_unfold() {
    // `<unfold(x), y> == <x, fold(y)>` for exact dyadic fixtures, in
    // both pairing directions of the same identity.
    let x = Tensor::new([8], (1..=8).map(|v| v as f64).collect::<Vec<_>>());
    let y = Tensor::new([3, 3], (1..=9).map(|v| v as f64).collect::<Vec<_>>());
    let unfolded_pairing = (x.unfold(0, 3, 2, 1) * y.clone()).sum();
    let folded_pairing = (x * y.fold(0, 3, 2, 1, 8)).sum();
    assert_eq!(unfolded_pairing.to_vec(), folded_pairing.to_vec());
}

#[test]
#[should_panic(expected = "disagrees with the 3 windows")]
fn fold_rejects_a_mismatched_window_count() {
    Tensor::filled([4, 3], 1.0_f64).fold(0, 3, 2, 1, 8);
}

#[test]
fn transcendentals_apply_elementwise() {
    let tensor = Tensor::new([2], [0.0_f64, 1.0]);
    let result = tensor.tanh();
    assert!((result.to_vec()[0]).abs() < 1e-12);
    assert!((result.to_vec()[1] - 1.0_f64.tanh()).abs() < 1e-12);
}

#[test]
fn tensor_payloads_flow_through_the_graph() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2], [0.0_f64, 1.0]));
    let y = x.tanh();

    let evaluation = network.forward();
    let result = evaluation.of(y);
    assert!((result.to_vec()[0]).abs() < 1e-12);
    assert!((result.to_vec()[1] - 1.0_f64.tanh()).abs() < 1e-12);
}

#[test]
fn engine_trains_tensor_payloads_unchanged() {
    // Two independent scalar problems, carried as one tensor of shape [2]:
    // fit `w * x = y` for `w = [5, -3]`. The engine is the same one that
    // trains scalar graphs; only the payload changed. The elementwise
    // squared errors are reduced with `sum` into the scalar target that
    // `backward` requires; the per-element gradients are unchanged since
    // the problems are independent.
    let network = Network::new();
    let w = network.parameter(Tensor::filled([2], 0.0_f64));
    let x = network.leaf(Tensor::new([2], [3.0, 2.0]));
    let y = network.leaf(Tensor::new([2], [15.0, -6.0]));

    let error = w * x - y;
    let loss = (error * error).sum();

    let w_symbol = w.symbol();
    let loss_symbol = loss.symbol();

    let learning_rate = Tensor::filled([2], 0.05);
    let mut network = network;
    for _ in 0..200 {
        let loss = network.resolve(loss_symbol);
        let evaluation = network.forward();
        let gradients = evaluation.backward(loss);
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.clone()
        });
    }

    let learned = network.resolve(w_symbol).payload().unwrap();
    assert!((learned.to_vec()[0] - 5.0).abs() < 1e-6);
    assert!((learned.to_vec()[1] + 3.0).abs() < 1e-6);
}

#[test]
fn matmul_transpose_and_sum_compute() {
    let matrix = Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]);
    let column = Tensor::new([2, 1], [5.0, 6.0]);

    let product = matrix.matmul(&column);
    assert_eq!(product.shape(), Shape::new([2, 1]));
    assert_eq!(product.to_vec(), &[17.0, 39.0]);

    let transposed = matrix.transpose();
    assert_eq!(transposed.to_vec(), &[1.0, 3.0, 2.0, 4.0]);

    let total = matrix.sum();
    assert_eq!(total.shape(), Shape::scalar());
    assert_eq!(total.to_vec(), &[10.0]);

    let spread = total.broadcast_like(&column);
    assert_eq!(spread.shape(), Shape::new([2, 1]));
    assert_eq!(spread.to_vec(), &[10.0, 10.0]);
}

#[test]
#[should_panic(expected = "inner dimensions")]
fn matmul_rejects_disagreeing_shapes() {
    let left = Tensor::new([2, 2], vec![1.0_f64; 4]);
    let right = Tensor::new([3, 1], vec![1.0_f64; 3]);
    left.matmul(&right);
}

#[test]
#[should_panic(expected = "single-element")]
fn broadcast_rejects_multi_element_sources() {
    let source = Tensor::new([2], [1.0_f64, 2.0]);
    let reference = Tensor::new([3], vec![0.0_f64; 3]);
    source.broadcast_like(&reference);
}

#[test]
fn matmul_routes_gradients_through_transposed_operands() {
    let network = Network::new();
    let a = network.leaf(Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]));
    let b = network.leaf(Tensor::new([2, 1], [5.0, 6.0]));

    let loss = a.matmul(b).sum();

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(loss), Tensor::new([], [56.0]));

    // With the loss seeded at one, `dA = 1 . B^T` row-repeated and
    // `dB = A^T . 1` column-summed.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(a).to_vec(), &[5.0, 6.0, 5.0, 6.0]);
    assert_eq!(gradients.of(b).to_vec(), &[4.0, 6.0]);
}

#[test]
fn broadcast_and_sum_are_adjoint() {
    let network = Network::new();
    let scalar = network.leaf(Tensor::new([], [2.0_f64]));
    let reference = network.leaf(Tensor::new([3], [1.0, 1.0, 1.0]));

    let loss = scalar.broadcast_like(reference).sum();

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(loss), Tensor::new([], [6.0]));

    // The broadcast spreads to three positions, so the scalar's gradient
    // is the sum of three ones; the shape reference receives none.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(scalar).to_vec(), &[3.0]);
    assert_eq!(gradients.of(reference).to_vec(), &[0.0, 0.0, 0.0]);
}

#[test]
#[should_panic(expected = "a recording panicked earlier")]
fn poisoned_network_names_its_cause() {
    let network = Network::new();
    let a = network.leaf(Tensor::new([2], [1.0_f64, 2.0]));
    let b = network.leaf(Tensor::new([3], [1.0, 2.0, 3.0]));

    // A caught shape mismatch poisons the tape; every later use fails
    // fatally, and the message must point at the recording panic, not
    // just the lock mechanics.
    let mismatch = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _ = a + b;
    }));
    assert!(mismatch.is_err());
    network.len();
}

#[test]
fn sum_along_reduces_the_named_axis() {
    let matrix = Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let columns = matrix.sum_along(0);
    assert_eq!(columns.shape(), Shape::new([3]));
    assert_eq!(columns.to_vec(), &[5.0, 7.0, 9.0]);

    let rows = matrix.sum_along(1);
    assert_eq!(rows.shape(), Shape::new([2]));
    assert_eq!(rows.to_vec(), &[6.0, 15.0]);
}

#[test]
fn maximum_picks_the_larger_element() {
    let left = Tensor::new([3], [1.0_f64, 5.0, -2.0]);
    let right = Tensor::new([3], [4.0, 2.0, -3.0]);
    assert_eq!(left.maximum(&right).to_vec(), &[4.0, 5.0, -2.0]);
}

#[test]
fn sqrt_applies_elementwise() {
    let tensor = Tensor::new([2], [4.0_f64, 9.0]);
    assert_eq!(tensor.sqrt().to_vec(), &[2.0, 3.0]);
}

#[test]
fn step_indicates_reached_thresholds() {
    let values = Tensor::new([3], [1.0_f64, 2.0, 3.0]);
    let thresholds = Tensor::filled([3], 2.0);
    assert_eq!(values.step(&thresholds).to_vec(), &[0.0, 1.0, 1.0]);
}

#[test]
fn max_along_reduces_the_named_axis() {
    let matrix = Tensor::new([2, 3], [1.0_f64, 5.0, 3.0, 4.0, 2.0, 6.0]);

    let columns = matrix.max_along(0);
    assert_eq!(columns.shape(), Shape::new([3]));
    assert_eq!(columns.to_vec(), &[4.0, 5.0, 6.0]);

    let rows = matrix.max_along(1);
    assert_eq!(rows.shape(), Shape::new([2]));
    assert_eq!(rows.to_vec(), &[5.0, 6.0]);
}

#[test]
#[should_panic(expected = "out of rank")]
fn max_along_rejects_excessive_axes() {
    Tensor::filled([2, 3], 1.0_f64).max_along(2);
}

#[test]
fn broadcast_along_repeats_the_named_axis() {
    let row = Tensor::new([3], [1.0_f64, 2.0, 3.0]);
    let reference = Tensor::filled([2, 3], 0.0);

    let spread = row.broadcast_along(0, &reference);
    assert_eq!(spread.shape(), Shape::new([2, 3]));
    assert_eq!(spread.to_vec(), &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);

    let column = Tensor::new([2], [1.0_f64, 2.0]);
    let spread = column.broadcast_along(1, &reference);
    assert_eq!(spread.to_vec(), &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0]);
}

#[test]
fn axis_sum_and_broadcast_are_adjoint() {
    let network = Network::new();
    let bias = network.leaf(Tensor::new([3], [1.0_f64, 2.0, 3.0]));
    let reference = network.leaf(Tensor::filled([2, 3], 0.0));

    let loss = bias.broadcast_along(0, reference).sum();

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(loss), Tensor::new([], [12.0]));

    // Each bias element is repeated across the two rows, so its
    // gradient is the sum of two ones; the shape reference gets none.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(bias).to_vec(), &[2.0, 2.0, 2.0]);
    assert_eq!(gradients.of(reference).to_vec(), &[0.0; 6]);
}

#[test]
#[should_panic(expected = "out of rank")]
fn sum_along_rejects_excessive_axes() {
    let network = Network::new();
    let matrix = network.leaf(Tensor::filled([2, 3], 1.0_f64));
    matrix.sum_along(2);
}

#[test]
#[should_panic(expected = "requires the remaining shape")]
fn broadcast_along_rejects_mismatched_operands() {
    let network = Network::new();
    let wrong = network.leaf(Tensor::new([2], [1.0_f64, 2.0]));
    let reference = network.leaf(Tensor::filled([2, 3], 0.0));
    wrong.broadcast_along(0, reference);
}

#[test]
#[should_panic(expected = "recorded shape")]
fn forward_with_rejects_mismatched_shapes() {
    let network = Network::new();
    let input = network.input(Tensor::new([2], [1.0_f64, 2.0]));
    network.forward_with([(input.symbol(), Tensor::new([3], [1.0, 2.0, 3.0]))]);
}

#[test]
#[should_panic(expected = "scalar target")]
fn backward_rejects_non_scalar_targets() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2], [1.0_f64, 2.0]));
    let doubled = x + x;

    let evaluation = network.forward();
    evaluation.backward(doubled);
}

#[test]
fn broadcast_restores_singleton_shapes_in_backward() {
    let network = Network::new();
    let source = network.leaf(Tensor::new([1], [2.0_f64]));
    let reference = network.leaf(Tensor::new([3], [1.0, 1.0, 1.0]));

    let loss = source.broadcast_like(reference).sum();

    let evaluation = network.forward();
    let gradients = evaluation.backward(loss);
    assert_eq!(*gradients.of(source), Tensor::new([1], [3.0]));
}

#[test]
#[should_panic(expected = "preserve the parameter's shape")]
fn update_rejects_shape_changing_rules() {
    let network = Network::new();
    let w = network.parameter(Tensor::new([1], [1.0_f64]));
    let loss = w.sum();

    let evaluation = network.forward();
    let gradients = evaluation.backward(loss);
    network.update(&gradients, |_parameter, _gradient| {
        Tensor::new([2], [7.0, 8.0])
    });
}

#[test]
fn linear_regression_trains_in_matrix_form() {
    // Fit `X . w = y` for `w = [[2], [-1]]`: the layer-sized problem that
    // took O(inputs * outputs) scalar nodes now takes a handful of tensor
    // nodes.
    let network = Network::new();
    let x = network.leaf(Tensor::new([3, 2], [1.0_f64, 0.0, 0.0, 1.0, 1.0, 1.0]));
    let y = network.leaf(Tensor::new([3, 1], [2.0, -1.0, 1.0]));
    let w = network.parameter(Tensor::filled([2, 1], 0.0_f64));

    let error = x.matmul(w) - y;
    let loss = (error * error).sum();

    let w_symbol = w.symbol();
    let loss_symbol = loss.symbol();

    let learning_rate = Tensor::new([], [0.05]);
    let mut network = network;
    for _ in 0..300 {
        let loss = network.resolve(loss_symbol);
        let evaluation = network.forward();
        let gradients = evaluation.backward(loss);
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    let learned = network.resolve(w_symbol).payload().unwrap();
    assert!((learned.to_vec()[0] - 2.0).abs() < 1e-6);
    assert!((learned.to_vec()[1] + 1.0).abs() < 1e-6);
}

#[test]
fn tensor_literals_mix_into_expressions() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2], [1.0_f64, 2.0]));

    let y = Tensor::filled([2], 10.0) * x + Tensor::filled([2], 1.0);

    let evaluation = network.forward();
    assert_eq!(evaluation.of(y).to_vec(), &[11.0, 21.0]);
}

#[test]
fn shapes_are_known_before_anything_runs() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([3, 2], vec![1.0_f64; 6]));
    let w = network.parameter(Tensor::filled([2, 1], 0.0_f64));

    let prediction = x.matmul(w);
    let loss = (prediction * prediction).sum();

    // No forward has run; the shapes were inferred at record time.
    assert_eq!(prediction.shape(), Shape::new([3, 1]));
    assert_eq!(loss.shape(), Shape::scalar());
}

#[test]
#[should_panic(expected = "matmul cannot multiply [2, 2] by [3, 1]")]
fn recording_rejects_disagreeing_matmul_shapes() {
    let network = Network::new();
    let a = network.leaf(Tensor::new([2, 2], vec![1.0_f64; 4]));
    let b = network.leaf(Tensor::new([3, 1], vec![1.0_f64; 3]));
    a.matmul(b);
}

#[test]
#[should_panic(expected = "equal shapes")]
fn recording_rejects_mismatched_addition() {
    let network = Network::new();
    let a = network.leaf(Tensor::new([2], vec![1.0_f64; 2]));
    let b = network.leaf(Tensor::new([3], vec![1.0_f64; 3]));
    let _ = a + b;
}

#[test]
#[should_panic(expected = "single-element operand")]
fn recording_rejects_broadcast_of_multi_element_sources() {
    let network = Network::new();
    let source = network.leaf(Tensor::new([2], vec![1.0_f64; 2]));
    let reference = network.leaf(Tensor::new([3], vec![0.0_f64; 3]));
    source.broadcast_like(reference);
}

#[test]
fn reshape_reinterprets_elements_in_logical_order() {
    let matrix = Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let flat = matrix.reshape(Shape::new([6]));
    assert_eq!(flat.shape(), Shape::new([6]));
    assert_eq!(flat.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let reshaped = matrix.reshape(Shape::new([3, 2]));
    assert_eq!(reshaped.shape(), Shape::new([3, 2]));
    assert_eq!(reshaped.to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn reshape_of_a_contiguous_tensor_shares_storage() {
    let matrix = Tensor::new([2, 3], vec![1.0_f64; 6]);
    let reshaped = matrix.reshape(Shape::new([6]));
    assert_eq!(
        matrix.as_slice().unwrap().as_ptr(),
        reshaped.as_slice().unwrap().as_ptr()
    );
}

#[test]
fn reshape_of_a_strided_view_materializes_in_order() {
    // A transpose is non-contiguous, so reshaping it copies into a fresh
    // contiguous buffer holding the elements in logical order.
    let matrix = Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let reshaped = matrix.transpose().reshape(Shape::new([6]));
    assert_eq!(reshaped.to_vec(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
}

#[test]
fn reshape_of_a_broadcast_view_keeps_the_view() {
    let row = Tensor::new([1, 3], [1.0_f64, 2.0, 3.0]);
    let like = Tensor::filled([5, 1, 3], 0.0);
    let squeezed = row.broadcast_along(0, &like).reshape(Shape::new([5, 3]));
    // The reshape drops only a unit axis, so the stride-0 view survives
    // instead of materializing a contiguous copy.
    assert!(squeezed.as_slice().is_none());
    assert_eq!(squeezed.to_vec(), [1.0, 2.0, 3.0].repeat(5));
}

#[test]
fn arithmetic_over_spread_views_matches_the_materialized_twins() {
    let matrix = Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let row_spread = Tensor::new([3], [10.0_f64, 20.0, 30.0]).broadcast_along(0, &matrix);
    let column_spread = Tensor::new([2], [100.0_f64, 200.0]).broadcast_along(1, &matrix);
    let materialized = |view: &Tensor<f64>| Tensor::new([2, 3], view.to_vec());

    // A unit-stride run against a stride-0 run, both ways around, and a
    // stride-0 pair: each takes the run path and must agree bitwise with
    // the same operation over materialized operands.
    assert_eq!(
        matrix.clone() + row_spread.clone(),
        matrix.clone() + materialized(&row_spread)
    );
    assert_eq!(
        column_spread.clone() * matrix.clone(),
        materialized(&column_spread) * matrix.clone()
    );
    assert_eq!(
        column_spread.clone() + row_spread.clone(),
        materialized(&column_spread) + materialized(&row_spread)
    );
    assert_eq!(
        row_spread.clone() * row_spread.clone(),
        materialized(&row_spread) * materialized(&row_spread)
    );
}

#[test]
fn arithmetic_over_transposed_views_matches_the_materialized_twins() {
    // Inner strides above one decline the run path; the odometer
    // fallback must still answer.
    let matrix = Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let transposed = matrix.transpose();
    let materialized = Tensor::new([3, 2], transposed.to_vec());
    assert_eq!(
        transposed.clone() + transposed.clone(),
        materialized.clone() + materialized
    );
}

#[test]
fn map_of_a_broadcast_view_keeps_the_view() {
    let row = Tensor::new([3], [1.0_f64, 2.0, 3.0]);
    let like = Tensor::filled([4, 3], 0.0);
    let spread = row.broadcast_along(0, &like);
    let negated = -spread.clone();
    // The map transforms the three distinct elements and keeps the
    // stride-0 layout instead of materializing twelve.
    assert!(negated.as_slice().is_none());
    assert_eq!(negated.to_vec(), [-1.0, -2.0, -3.0].repeat(4));

    let exponentials = spread.exp();
    assert!(exponentials.as_slice().is_none());
    assert_eq!(exponentials.to_vec(), row.exp().to_vec().repeat(4));
}

#[test]
fn map_of_a_narrow_sliver_falls_back_to_the_logical_walk() {
    // Narrowing the inner axis leaves a window wider than the volume, so
    // the map materializes in logical order instead of walking it.
    let matrix = Tensor::new(
        [3, 4],
        (0..12).map(|index| index as f64).collect::<Vec<_>>(),
    );
    let sliver = matrix.narrow(1, 1, 2);
    let negated = -sliver;
    assert!(negated.as_slice().is_some());
    assert_eq!(negated.to_vec(), vec![-1.0, -2.0, -5.0, -6.0, -9.0, -10.0]);
}

#[test]
fn zip_of_a_spread_view_and_a_constant_keeps_the_view() {
    let row = Tensor::new([3], [1.0_f64, 2.0, 3.0]);
    let like = Tensor::filled([4, 3], 0.0);
    let spread = row.broadcast_along(0, &like);
    let shifted = spread + Tensor::filled([4, 3], 10.0);
    assert!(shifted.as_slice().is_none());
    assert_eq!(shifted.to_vec(), [11.0, 12.0, 13.0].repeat(4));
}

#[test]
#[should_panic(expected = "changes the number of elements")]
fn reshape_rejects_volume_changes() {
    Tensor::new([2, 3], vec![1.0_f64; 6]).reshape(Shape::new([2, 2]));
}

#[test]
fn permute_reorders_axes() {
    let tensor = Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let permuted = tensor.permute(&[1, 0]);
    assert_eq!(permuted.shape(), Shape::new([3, 2]));
    // For a rank-2 tensor a permutation of the axes is a transpose.
    assert_eq!(permuted.to_vec(), tensor.transpose().to_vec());
}

#[test]
fn permute_is_rank_general() {
    let tensor = Tensor::new([2, 1, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let permuted = tensor.permute(&[2, 0, 1]);
    assert_eq!(permuted.shape(), Shape::new([3, 2, 1]));
}

#[test]
#[should_panic(expected = "repeats axis")]
fn permute_rejects_non_permutations() {
    Tensor::new([2, 3], vec![1.0_f64; 6]).permute(&[0, 0]);
}

#[test]
fn reshape_routes_gradients_back() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let weight = network.leaf(Tensor::new([6], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]));

    // Weighting each element of the flattened view by a distinct factor makes
    // the gradient the weights reshaped back to `x`'s shape.
    let loss = (x.reshape([6]) * weight).sum();
    let gradients = network.forward().backward(loss);
    assert_eq!(gradients.of(x).shape(), Shape::new([2, 3]));
    assert_eq!(
        gradients.of(x).to_vec(),
        vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    );
}

#[test]
fn permute_routes_gradients_back() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    let weight = network.leaf(Tensor::new([3, 2], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]));

    // `x.permute([1, 0])` transposes, so weight `(i, j)` multiplies
    // `x(j, i)`; the gradient is the weights permuted back to `x`'s shape.
    let loss = (x.permute([1, 0]) * weight).sum();
    let gradients = network.forward().backward(loss);
    assert_eq!(gradients.of(x).shape(), Shape::new([2, 3]));
    assert_eq!(
        gradients.of(x).to_vec(),
        vec![10.0, 30.0, 50.0, 20.0, 40.0, 60.0]
    );
}

#[test]
fn squeeze_and_unsqueeze_adjust_extent_one_axes() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([3], [1.0_f64, 2.0, 3.0]));
    let unsqueezed = x.unsqueeze(0);
    let squeezed = unsqueezed.squeeze(0);
    assert_eq!(unsqueezed.shape(), Shape::new([1, 3]));
    assert_eq!(squeezed.shape(), Shape::new([3]));

    let evaluation = network.forward();
    assert_eq!(evaluation.of(unsqueezed).to_vec(), vec![1.0, 2.0, 3.0]);
    assert_eq!(evaluation.of(squeezed).to_vec(), vec![1.0, 2.0, 3.0]);
}

#[test]
#[should_panic(expected = "changes the number of elements")]
fn recording_rejects_volume_changing_reshape() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2, 3], vec![1.0_f64; 6]));
    x.reshape([4]);
}

#[test]
fn narrow_selects_a_window_along_an_axis() {
    let matrix = Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);

    let columns = matrix.narrow(1, 1, 2);
    assert_eq!(columns.shape(), Shape::new([2, 2]));
    assert_eq!(columns.to_vec(), vec![2.0, 3.0, 5.0, 6.0]);

    let row = matrix.narrow(0, 1, 1);
    assert_eq!(row.shape(), Shape::new([1, 3]));
    assert_eq!(row.to_vec(), vec![4.0, 5.0, 6.0]);
}

#[test]
fn narrow_of_the_outer_axis_stays_contiguous() {
    // A window over whole rows keeps the inner axis contiguous, so it can
    // still expose a borrowed slice of the shared buffer.
    let matrix = Tensor::new([3, 2], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let middle = matrix.narrow(0, 1, 1);
    assert_eq!(middle.as_slice().unwrap().to_vec(), vec![3.0, 4.0]);
}

#[test]
#[should_panic(expected = "exceeds axis")]
fn narrow_rejects_windows_past_the_axis() {
    Tensor::new([2, 3], vec![1.0_f64; 6]).narrow(1, 2, 2);
}

#[test]
#[should_panic(expected = "at least one element")]
fn narrow_rejects_empty_windows() {
    Tensor::new([2, 3], vec![1.0_f64; 6]).narrow(1, 0, 0);
}

#[test]
#[should_panic(expected = "at least one element")]
fn narrow_rejects_empty_windows_at_the_axis_end() {
    Tensor::new([2, 3], vec![1.0_f64; 6]).narrow(0, 2, 0);
}

#[test]
#[should_panic(expected = "at least one element")]
fn narrow_rejects_empty_windows_at_recording() {
    let network = Network::new();
    let matrix = network.leaf(Tensor::new([2, 3], vec![1.0_f64; 6]));
    matrix.narrow(1, 0, 0);
}

#[test]
#[should_panic(expected = "overflows")]
fn narrow_rejects_overflowing_windows() {
    Tensor::new([2, 3], vec![1.0_f64; 6]).narrow(1, usize::MAX, 2);
}

#[test]
#[should_panic(expected = "overflows")]
fn pad_rejects_overflowing_windows() {
    Tensor::new([1], [5.0_f64]).pad(0, usize::MAX, 2);
}

#[test]
fn pad_places_a_window_into_zeros() {
    let window = Tensor::new([2, 2], [2.0_f64, 3.0, 5.0, 6.0]);
    let padded = window.pad(1, 1, 3);
    assert_eq!(padded.shape(), Shape::new([2, 3]));
    assert_eq!(padded.to_vec(), vec![0.0, 2.0, 3.0, 0.0, 5.0, 6.0]);
}

#[test]
fn narrow_routes_gradients_to_the_window() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2, 3], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));

    // Summing columns 1..3 gives a gradient of one there and zero in the
    // column the window excludes.
    let loss = x.narrow(1, 1, 2).sum();
    let gradients = network.forward().backward(loss);
    assert_eq!(gradients.of(x).shape(), Shape::new([2, 3]));
    assert_eq!(gradients.of(x).to_vec(), vec![0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
}

#[test]
fn embedding_lookup_is_a_one_hot_matmul() {
    // An embedding lookup `table[tokens]` is `onehot.matmul(table)`, so it
    // needs no dedicated gather op. The one-hot rows are per-run data fed as
    // an input, so one recorded graph serves any minibatch, and `matmul`'s
    // backward is exactly the scatter-add embedding gradient.
    let network = Network::new();
    let table = network.leaf(Tensor::new([3, 2], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    // Only the shape of the token batch is fixed at record time; the tokens
    // themselves arrive per run through `forward_with`.
    let onehot = network.input(Tensor::filled([3, 3], 0.0));
    let onehot_symbol = onehot.symbol();

    let embedded = onehot.matmul(table);
    let loss = embedded.sum();

    // Feed the tokens [0, 2, 0] as one-hot rows over a vocabulary of three.
    let tokens = Tensor::new([3, 3], [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
    let evaluation = network.forward_with([(onehot_symbol, tokens)]);

    // The result rows are the looked-up table rows, in token order.
    assert_eq!(
        evaluation.of(embedded).to_vec(),
        vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0]
    );

    // Token 0 is selected twice, so its row accumulates two ones; token 1 is
    // never selected, so its row's gradient is zero. That accumulation is the
    // scatter-add a dedicated gather would have to implement by hand.
    let gradients = evaluation.backward(loss);
    assert_eq!(
        gradients.of(table).to_vec(),
        vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]
    );
}

#[test]
fn selection_is_a_one_hot_matrix() {
    let selection = Tensor::selection(vec![0usize, 2, 0], 3, 1.0_f64);
    assert_eq!(selection.shape(), Shape::new([3, 3]));
    assert_eq!(
        selection.to_vec(),
        vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]
    );
}

#[test]
#[should_panic(expected = "out of vocabulary")]
fn selection_rejects_out_of_range_indices() {
    Tensor::selection(vec![3usize], 3, 1.0_f64);
}

#[test]
fn gather_selects_table_rows() {
    let table = Tensor::new([3, 2], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let selection = Tensor::selection(vec![0usize, 2, 0], 3, 1.0);
    let gathered = table.gather(&selection);
    assert_eq!(gathered.shape(), Shape::new([3, 2]));
    assert_eq!(gathered.to_vec(), vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0]);
}

#[test]
fn scatter_accumulates_repeated_rows() {
    let gradient = Tensor::filled([3, 2], 1.0_f64);
    let selection = Tensor::selection(vec![0usize, 2, 0], 3, 1.0);

    // Rows are scattered by index; token 0 is selected twice, so its row
    // accumulates two ones, and token 1 (never selected) stays zero.
    let scattered = gradient.scatter(&selection, 3);
    assert_eq!(scattered.shape(), Shape::new([3, 2]));
    assert_eq!(scattered.to_vec(), vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]);
}

#[test]
#[should_panic(expected = "disagree with the selection count")]
fn scatter_rejects_extra_gradient_rows() {
    let gradient = Tensor::new([2, 1], [10.0_f64, 20.0]);
    let selection = Tensor::selection([0_usize], 2, 1.0);
    gradient.scatter(&selection, 2);
}

#[test]
#[should_panic(expected = "disagree with the selection count")]
fn scatter_rejects_missing_gradient_rows() {
    let gradient = Tensor::new([1, 1], [10.0_f64]);
    let selection = Tensor::selection([0_usize, 1], 2, 1.0);
    gradient.scatter(&selection, 2);
}

#[test]
#[should_panic(expected = "disagree with the selection vocabulary")]
fn scatter_rejects_a_foreign_vocabulary() {
    let gradient = Tensor::new([1, 1], [10.0_f64]);
    let selection = Tensor::selection([0_usize], 3, 1.0);
    gradient.scatter(&selection, 2);
}

#[test]
#[should_panic(expected = "leading selection axis")]
fn scatter_rejects_a_rank_zero_gradient() {
    let gradient = Tensor::filled([1], 10.0_f64).sum();
    let selection = Tensor::selection([0_usize], 1, 1.0);
    gradient.scatter(&selection, 1);
}

#[test]
fn selection_densifies_for_non_gather_operations() {
    // A selection is stored as its indices, but any operation other than
    // gather still works by densifying it to the one-hot it represents.
    let selection = Tensor::selection(vec![1usize, 1], 3, 1.0_f64);
    let transposed = selection.transpose();
    assert_eq!(transposed.shape(), Shape::new([3, 2]));
    assert_eq!(transposed.to_vec(), vec![0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
}

#[test]
fn gather_op_routes_gradients_by_scatter_add() {
    let network = Network::new();
    let table = network.leaf(Tensor::new([3, 2], [1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]));
    // The selection is a per-run input: only its shape is fixed at record
    // time, so one graph serves any batch of tokens.
    let selection = network.input(Tensor::selection(vec![0usize, 0, 0], 3, 1.0));
    let selection_symbol = selection.symbol();

    let embedded = table.gather(selection);
    let loss = embedded.sum();

    let evaluation = network.forward_with([(
        selection_symbol,
        Tensor::selection(vec![0usize, 2, 0], 3, 1.0),
    )]);
    assert_eq!(
        evaluation.of(embedded).to_vec(),
        vec![1.0, 2.0, 5.0, 6.0, 1.0, 2.0]
    );

    // The dedicated op's backward is the scatter-add, with no term for the
    // selection at all: the indices are data.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(table).shape(), Shape::new([3, 2]));
    assert_eq!(
        gradients.of(table).to_vec(),
        vec![2.0, 2.0, 0.0, 0.0, 1.0, 1.0]
    );
    assert_eq!(gradients.of(selection).to_vec(), vec![0.0; 9]);
}

#[test]
fn gather_infers_the_result_shape() {
    let network = Network::new();
    let table = network.leaf(Tensor::new([4, 3], vec![0.0_f64; 12]));
    let selection = network.input(Tensor::selection(vec![0usize, 1], 4, 1.0));

    let embedded = table.gather(selection);
    // [count, vocab] gather [vocab, dim] -> [count, dim].
    assert_eq!(embedded.shape(), Shape::new([2, 3]));
}

#[test]
#[should_panic(expected = "does not match table rows")]
fn gather_rejects_vocabulary_mismatch() {
    let network = Network::new();
    let table = network.leaf(Tensor::new([3, 2], vec![0.0_f64; 6]));
    let selection = network.input(Tensor::selection(vec![0usize], 4, 1.0));
    table.gather(selection);
}

#[test]
fn log_softmax_normalizes_along_the_named_axis() {
    let network = Network::new();
    let logits = network.leaf(Tensor::new([2, 2], [0.0_f64, 0.0, 1.0, 3.0]));
    let log_probabilities = logits.log_softmax(1);
    assert_eq!(log_probabilities.shape(), Shape::new([2, 2]));

    let evaluation = network.forward();
    let probabilities = evaluation.of(log_probabilities).exp();
    for total in probabilities.sum_along(1).to_vec() {
        assert!((total - 1.0).abs() < 1e-12);
    }
}

#[test]
fn log_softmax_routes_gradients_through_the_probabilities() {
    let network = Network::new();
    let logits = network.leaf(Tensor::new([1, 2], [0.0_f64, 3.0_f64.ln()]));

    // Summing one row of log-probabilities seeds every class with one, so
    // the cotangent is `1 - classes * softmax`: `[1 - 2 * 0.25, 1 - 2 * 0.75]`.
    let loss = logits.log_softmax(1).sum();

    let evaluation = network.forward();
    let gradients = evaluation.backward(loss);
    let expected = [0.5, -0.5];
    for (computed, expected) in gradients.of(logits).to_vec().into_iter().zip(expected) {
        assert!((computed - expected).abs() < 1e-12);
    }
}

#[test]
#[should_panic(expected = "out of rank")]
fn log_softmax_rejects_excessive_axes() {
    let network = Network::new();
    let logits = network.leaf(Tensor::filled([2, 3], 0.0_f64));
    logits.log_softmax(2);
}

#[test]
fn relu_masks_gradients_by_sign() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([4], [-2.0_f64, -0.5, 0.0, 3.0]));
    let activated = x.relu();
    let loss = activated.sum();

    let evaluation = network.forward();
    assert_eq!(evaluation.of(activated).to_vec(), &[0.0, 0.0, 0.0, 3.0]);

    // The gradient passes only where the operand reached zero; the
    // subgradient at zero itself is one.
    let gradients = evaluation.backward(loss);
    assert_eq!(gradients.of(x).to_vec(), &[0.0, 0.0, 1.0, 1.0]);
}

#[test]
fn roots_and_powers_route_gradients() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2], [4.0_f64, 9.0]));
    let exponent = network.leaf(Tensor::filled([2], 2.0));
    let loss = (x.sqrt() + x.powf(exponent)).sum();

    let evaluation = network.forward();
    assert_eq!(*evaluation.of(loss), Tensor::new([], [102.0]));

    // Per element: `1 / (2 sqrt(x)) + 2 x`, so `[0.25 + 8, 1/6 + 18]`.
    let gradients = evaluation.backward(loss);
    let expected = [8.25, 18.0 + 1.0 / 6.0];
    for (computed, expected) in gradients.of(x).to_vec().into_iter().zip(expected) {
        assert!((computed - expected).abs() < 1e-12);
    }
}

#[test]
#[should_panic(expected = "equal shapes")]
fn maximum_rejects_mismatched_operands() {
    let network = Network::new();
    let left = network.leaf(Tensor::filled([2], 0.0_f64));
    let right = network.leaf(Tensor::filled([3], 0.0));
    left.maximum(right);
}

#[test]
fn elementwise_lanes_agree_bitwise() {
    // The slice fast lanes must hand the combiner the same pairs in
    // the same order as the logical-order iterators: a strided view
    // (the generic lane) of the same logical values answers
    // identically, bit for bit.
    let elements: Vec<f64> = (0..64).map(|index| (index as f64 - 31.5) / 7.0).collect();
    let contiguous = Tensor::new([8, 8], elements.clone());
    let mut transposed_elements = vec![0.0_f64; 64];
    for row in 0..8 {
        for column in 0..8 {
            transposed_elements[column * 8 + row] = elements[row * 8 + column];
        }
    }
    let view = Tensor::new([8, 8], transposed_elements).transpose();

    let bits = |tensor: &Tensor<f64>| -> Vec<u64> {
        tensor
            .to_vec()
            .iter()
            .map(|value| value.to_bits())
            .collect()
    };
    let product_fast = contiguous.clone() * contiguous.clone();
    let product_generic = view.clone() * view.clone();
    assert_eq!(bits(&product_fast), bits(&product_generic));

    let zero = Tensor::filled([8, 8], 0.0_f64);
    let seeded_fast = zero.clone() + contiguous.clone();
    let seeded_generic = zero + view;
    assert_eq!(bits(&seeded_fast), bits(&seeded_generic));
}

#[test]
fn windowed_product_matches_the_composed_reference() {
    // The fast patch fill against the composed formula, bitwise, over
    // padding and stride variants.
    use crate::payload::tensorial::composed_windowed_patches;

    let input = Tensor::new(
        [2, 3, 5, 4],
        (0..120)
            .map(|v| (v as f64) * 0.37 - 20.0)
            .collect::<Vec<_>>(),
    );
    let kernel = Tensor::new(
        [12, 4],
        (0..48).map(|v| (v as f64) * 0.11 - 2.0).collect::<Vec<_>>(),
    );
    for (stride, padding) in [(1, 0), (1, 1), (2, 0), (2, 1)] {
        let fast = input.windowed_product(&kernel, 2, 2, stride, padding);
        let composed = composed_windowed_patches(&input, 2, 2, stride, padding).matmul(&kernel);
        assert_eq!(
            fast.shape(),
            composed.shape(),
            "stride {stride} pad {padding}"
        );
        assert_eq!(
            fast.to_vec(),
            composed.to_vec(),
            "stride {stride} pad {padding}"
        );
    }
}

#[test]
fn windowed_product_falls_back_for_strided_views() {
    use crate::payload::tensorial::composed_windowed_patches;

    let base = Tensor::new(
        [2, 3, 5, 6],
        (0..180)
            .map(|v| (v as f64) * 0.19 - 15.0)
            .collect::<Vec<_>>(),
    );
    // A narrowed view is not contiguous: the fast fill must decline and
    // the fallback must agree with the reference anyway.
    let view = base.narrow(3, 1, 4);
    assert!(view.as_slice().is_none());
    let kernel = Tensor::new(
        [27, 2],
        (0..54).map(|v| (v as f64) * 0.07 - 1.5).collect::<Vec<_>>(),
    );
    let fast = view.windowed_product(&kernel, 3, 3, 1, 1);
    let composed = composed_windowed_patches(&view, 3, 3, 1, 1).matmul(&kernel);
    assert_eq!(fast.to_vec(), composed.to_vec());
}
