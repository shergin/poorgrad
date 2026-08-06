use crate::Shape;

use super::{Emittable, dense_literal, tensor_type};

#[test]
fn finite_literals_carry_a_dot() {
    assert_eq!(1.0_f32.literal(), "1.0");
    assert_eq!((-0.5_f64).literal(), "-0.5");
    // Rust's shortest form prints a bare mantissa here; MLIR requires
    // the dot.
    assert_eq!(1.0e-5_f32.literal(), "1.0e-5");
}

#[test]
fn non_finite_literals_print_as_bit_patterns() {
    assert_eq!(f32::NEG_INFINITY.literal(), "0xFF800000");
    assert_eq!(f32::INFINITY.literal(), "0x7F800000");
    assert_eq!(f64::NEG_INFINITY.literal(), "0xFFF0000000000000");
}

#[test]
fn tensor_types_follow_the_shape() {
    assert_eq!(tensor_type::<f32>(&Shape::scalar()), "tensor<f32>");
    assert_eq!(tensor_type::<f64>(&Shape::new([2, 3])), "tensor<2x3xf64>");
}

#[test]
fn dense_literals_splat_and_nest() {
    assert_eq!(
        dense_literal(&Shape::new([2, 2]), &[7.0_f32; 4]),
        "dense<7.0>"
    );
    assert_eq!(
        dense_literal(&Shape::new([2, 2]), &[1.0_f32, 2.0, 3.0, 4.0]),
        "dense<[[1.0, 2.0], [3.0, 4.0]]>"
    );
    assert_eq!(dense_literal(&Shape::scalar(), &[3.0_f64]), "dense<3.0>");
}
