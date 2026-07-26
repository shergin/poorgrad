use crate::{Differentiable, Elementary, Network};

use super::Tensor;

#[test]
fn new_builds_from_shape_and_elements() {
    let tensor = Tensor::new([2, 3], vec![1.0_f64; 6]);
    assert_eq!(tensor.shape(), &[2, 3]);
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
    assert_eq!(zero.shape(), &[2, 2]);
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
    let result = evaluation.value(y);
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

    let error = w * x + -y;
    let loss = error * error;

    let w_symbol = w.symbol();
    let loss_symbol = loss.symbol();

    let learning_rate = Tensor::filled([2], 0.05);
    let mut network = network;
    for _ in 0..200 {
        let loss = network.resolve(loss_symbol).unwrap();
        let evaluation = network.forward();
        let gradients = network.backward(&evaluation, loss);
        network = network.updated(gradients.as_field(), |parameter, gradient| {
            parameter.clone() + -(gradient.clone() * learning_rate.clone())
        });
    }

    let learned = network.resolve(w_symbol).unwrap().data().unwrap();
    assert!((learned.elements()[0] - 5.0).abs() < 1e-6);
    assert!((learned.elements()[1] + 3.0).abs() < 1e-6);
}
