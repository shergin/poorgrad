use crate::{Differentiable, Elementary, Network, Shape, Tensorial};

use super::Tensor;

#[test]
fn new_builds_from_shape_and_elements() {
    let tensor = Tensor::new([2, 3], vec![1.0_f64; 6]);
    assert_eq!(tensor.shape(), Shape::new([2, 3]));
    assert_eq!(tensor.elements().len(), 6);
}

#[test]
#[should_panic(expected = "shape does not match")]
fn new_rejects_mismatched_volume() {
    Tensor::new([2, 3], vec![1.0_f64; 5]);
}

#[test]
fn clone_shares_storage() {
    let tensor = Tensor::new([2], [1.0_f64, 2.0]);
    let clone = tensor.clone();
    assert!(tensor.elements().as_ptr() == clone.elements().as_ptr());
}

#[test]
fn arithmetic_applies_elementwise() {
    let left = Tensor::new([2], [1.0_f64, 2.0]);
    let right = Tensor::new([2], [10.0, 20.0]);

    assert_eq!((left.clone() + right.clone()).elements(), &[11.0, 22.0]);
    assert_eq!((right.clone() - left.clone()).elements(), &[9.0, 18.0]);
    assert_eq!((left.clone() * right.clone()).elements(), &[10.0, 40.0]);
    assert_eq!((right.clone() / left.clone()).elements(), &[10.0, 10.0]);
    assert_eq!((-left).elements(), &[-1.0, -2.0]);
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
    assert_eq!(zero.elements(), &[0.0; 4]);
    assert_eq!(one.elements(), &[1.0; 4]);
}

#[test]
fn transcendentals_apply_elementwise() {
    let tensor = Tensor::new([2], [0.0_f64, 1.0]);
    let result = tensor.tanh();
    assert!((result.elements()[0]).abs() < 1e-12);
    assert!((result.elements()[1] - 1.0_f64.tanh()).abs() < 1e-12);
}

#[test]
fn tensor_payloads_flow_through_the_graph() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2], [0.0_f64, 1.0]));
    let y = x.tanh();

    let evaluation = network.forward();
    let result = evaluation.of(y);
    assert!((result.elements()[0]).abs() < 1e-12);
    assert!((result.elements()[1] - 1.0_f64.tanh()).abs() < 1e-12);
}

#[test]
fn engine_trains_tensor_payloads_unchanged() {
    // Two independent scalar problems, carried as one tensor of shape [2]:
    // fit `w * x = y` for `w = [5, -3]`. The engine is the same one that
    // trains scalar graphs; only the payload changed.
    let network = Network::new();
    let w = network.parameter(Tensor::filled([2], 0.0_f64));
    let x = network.leaf(Tensor::new([2], [3.0, 2.0]));
    let y = network.leaf(Tensor::new([2], [15.0, -6.0]));

    let error = w * x - y;
    let loss = error * error;

    let w_symbol = w.symbol();
    let loss_symbol = loss.symbol();

    let learning_rate = Tensor::filled([2], 0.05);
    let mut network = network;
    for _ in 0..200 {
        let loss = network.resolve(loss_symbol);
        let evaluation = network.forward();
        let gradients = evaluation.backward(loss);
        network = network.updated(gradients.as_field(), |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.clone()
        });
    }

    let learned = network.resolve(w_symbol).data().unwrap();
    assert!((learned.elements()[0] - 5.0).abs() < 1e-6);
    assert!((learned.elements()[1] + 3.0).abs() < 1e-6);
}

#[test]
fn matmul_transpose_and_sum_compute() {
    let matrix = Tensor::new([2, 2], [1.0_f64, 2.0, 3.0, 4.0]);
    let column = Tensor::new([2, 1], [5.0, 6.0]);

    let product = matrix.matmul(&column);
    assert_eq!(product.shape(), Shape::new([2, 1]));
    assert_eq!(product.elements(), &[17.0, 39.0]);

    let transposed = matrix.transposed();
    assert_eq!(transposed.elements(), &[1.0, 3.0, 2.0, 4.0]);

    let total = matrix.sum();
    assert_eq!(total.shape(), Shape::scalar());
    assert_eq!(total.elements(), &[10.0]);

    let spread = total.broadcast_like(&column);
    assert_eq!(spread.shape(), Shape::new([2, 1]));
    assert_eq!(spread.elements(), &[10.0, 10.0]);
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
    assert_eq!(gradients.of(a).elements(), &[5.0, 6.0, 5.0, 6.0]);
    assert_eq!(gradients.of(b).elements(), &[4.0, 6.0]);
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
    assert_eq!(gradients.of(scalar).elements(), &[3.0]);
    assert_eq!(gradients.of(reference).elements(), &[0.0, 0.0, 0.0]);
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
fn updated_rejects_shape_changing_updates() {
    let network = Network::new();
    let w = network.parameter(Tensor::new([1], [1.0_f64]));
    let loss = w.sum();

    let evaluation = network.forward();
    let gradients = evaluation.backward(loss);
    network.updated(gradients.as_field(), |_parameter, _gradient| {
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
        network = network.updated(gradients.as_field(), |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    let learned = network.resolve(w_symbol).data().unwrap();
    assert!((learned.elements()[0] - 2.0).abs() < 1e-6);
    assert!((learned.elements()[1] + 1.0).abs() < 1e-6);
}

#[test]
fn tensor_literals_mix_into_expressions() {
    let network = Network::new();
    let x = network.leaf(Tensor::new([2], [1.0_f64, 2.0]));

    let y = Tensor::filled([2], 10.0) * x + Tensor::filled([2], 1.0);

    let evaluation = network.forward();
    assert_eq!(evaluation.of(y).elements(), &[11.0, 21.0]);
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
