use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::{Differentiable, Elementary, Shape, Tensor, Tensorial};

use super::GemmTask;

/// Computes the product over materialized operands in the logical
/// path's exact accumulation order — ascending inner index, seeded
/// from the first term — as the bit-level reference for every case.
fn reference<Element: Differentiable>(
    left: &Tensor<Element>,
    right: &Tensor<Element>,
) -> Vec<Element> {
    let a = left.to_vec();
    let b = right.to_vec();
    let rows = left.shape().axes()[0];
    let inner = left.shape().axes()[1];
    let columns = right.shape().axes()[1];
    let mut elements = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            let mut total = a[row * inner].clone() * b[column].clone();
            for step in 1..inner {
                total = total + a[row * inner + step].clone() * b[step * columns + column].clone();
            }
            elements.push(total);
        }
    }
    elements
}

/// Asserts that the product of the given operands equals the
/// logical-order reference bit for bit.
fn assert_matches_reference_f64(left: &Tensor<f64>, right: &Tensor<f64>) {
    let product = left.matmul(right);
    let expected: Vec<u64> = reference(left, right).iter().map(|e| e.to_bits()).collect();
    let actual: Vec<u64> = product.to_vec().iter().map(|e| e.to_bits()).collect();
    assert_eq!(actual, expected);
}

/// The `f32` twin of [`assert_matches_reference_f64`].
fn assert_matches_reference_f32(left: &Tensor<f32>, right: &Tensor<f32>) {
    let product = left.matmul(right);
    let expected: Vec<u32> = reference(left, right).iter().map(|e| e.to_bits()).collect();
    let actual: Vec<u32> = product.to_vec().iter().map(|e| e.to_bits()).collect();
    assert_eq!(actual, expected);
}

/// Builds a `[rows, columns]` tensor with distinct, sign-varied values.
fn varied(rows: usize, columns: usize, seed: i64) -> Tensor<f64> {
    let elements: Vec<f64> = (0..(rows * columns) as i64)
        .map(|index| ((index * 7 + seed * 13) % 23 - 11) as f64 / 4.0)
        .collect();
    Tensor::new([rows, columns], elements)
}

#[test]
fn contiguous_operands_match_the_logical_order() {
    for (rows, inner, columns) in [
        (1, 1, 1),
        (1, 4, 1),
        (5, 1, 3),
        (2, 3, 4),
        (8, 8, 8),
        (3, 17, 5),
    ] {
        let left = varied(rows, inner, 1);
        let right = varied(inner, columns, 2);
        assert_matches_reference_f64(&left, &right);
    }
}

#[test]
fn transposed_views_match_the_logical_order() {
    let left = varied(7, 4, 1).transpose();
    let right = varied(6, 7, 2).transpose();
    let plain_left = varied(4, 7, 3);
    let plain_right = varied(7, 6, 4);
    assert_matches_reference_f64(&left, &plain_right);
    assert_matches_reference_f64(&plain_left, &right);
    assert_matches_reference_f64(&left, &right);
}

#[test]
fn narrowed_windows_match_the_logical_order() {
    let left = varied(5, 9, 1).narrow(1, 2, 4);
    let right = varied(4, 8, 2).narrow(1, 3, 3);
    assert_matches_reference_f64(&left, &right);
    let tall = varied(9, 4, 3).narrow(0, 1, 5);
    assert_matches_reference_f64(&tall, &varied(4, 2, 4));
}

#[test]
fn broadcast_views_match_the_logical_order() {
    let reference_shape = varied(3, 4, 0);
    let left = varied(1, 4, 1)
        .reshape([4].into())
        .broadcast_along(0, &reference_shape);
    assert_matches_reference_f64(&left, &varied(4, 2, 2));
    let right_reference = varied(4, 5, 0);
    let right = varied(1, 4, 3)
        .reshape([4].into())
        .broadcast_along(1, &right_reference);
    assert_matches_reference_f64(&varied(2, 4, 4), &right);
}

#[test]
fn constant_operands_match_the_logical_order() {
    let constant = Tensor::filled([3, 4], 1.5_f64);
    let dense = varied(4, 2, 1);
    assert_matches_reference_f64(&constant, &dense);
    assert_matches_reference_f64(&varied(2, 3, 2), &Tensor::filled([3, 5], -0.25));
}

#[test]
fn f32_operands_match_the_logical_order() {
    let left = Tensor::new(
        [3, 5],
        (0..15).map(|i| (i as f32 - 7.0) / 3.0).collect::<Vec<_>>(),
    );
    let right = Tensor::new(
        [5, 2],
        (0..10).map(|i| (i as f32 - 4.0) / 7.0).collect::<Vec<_>>(),
    );
    assert_matches_reference_f32(&left, &right);
    assert_matches_reference_f32(&left, &Tensor::new([2, 5], vec![0.5_f32; 10]).transpose());
}

#[test]
fn negative_zero_terms_keep_their_sign() {
    // A product whose every term is `-0.0` must stay `-0.0`: seeding
    // from the first term preserves it where a zero-initialized
    // accumulator would answer `+0.0`.
    let left = Tensor::new([1, 2], [-0.0_f64, 0.0]);
    let right = Tensor::new([2, 1], [0.0_f64, -0.0]);
    let product = left.matmul(&right);
    assert_eq!(product.to_vec()[0].to_bits(), (-0.0_f64).to_bits());
    assert_matches_reference_f64(&left, &right);
}

#[test]
#[should_panic(expected = "does not span")]
fn task_rejects_a_short_left_operand() {
    GemmTask::new(&[1.0_f64; 3], [2, 1], &[1.0_f64; 4], [2, 1], 2, 2, 2);
}

#[test]
#[should_panic(expected = "does not span")]
fn task_rejects_a_short_right_operand() {
    GemmTask::new(&[1.0_f64; 4], [2, 1], &[1.0_f64; 3], [2, 1], 2, 2, 2);
}

#[test]
fn task_reports_its_dimensions_and_strides() {
    let a = [1.0_f64; 6];
    let b = [1.0_f64; 8];
    let task = GemmTask::new(&a, [3, 1], &b, [1, 3], 2, 3, 2);
    assert_eq!((task.m(), task.k(), task.n()), (2, 3, 2));
    assert_eq!(task.a_strides(), [3, 1]);
    assert_eq!(task.b_strides(), [1, 3]);
    assert_eq!(task.a().len(), 6);
    assert_eq!(task.b().len(), 8);
}

/// A scalar payload whose `gemm` seam answers a sentinel, proving
/// that `matmul` consults the element's seam before the built-in
/// kernels.
#[derive(Debug, Clone, PartialEq)]
struct Probe(f64);

impl Add for Probe {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Probe(self.0 + rhs.0)
    }
}

impl Sub for Probe {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Probe(self.0 - rhs.0)
    }
}

impl Mul for Probe {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Probe(self.0 * rhs.0)
    }
}

impl Div for Probe {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Probe(self.0 / rhs.0)
    }
}

impl Neg for Probe {
    type Output = Self;
    fn neg(self) -> Self {
        Probe(-self.0)
    }
}

impl Differentiable for Probe {
    fn zero_like(&self) -> Self {
        Probe(0.0)
    }
    fn one_like(&self) -> Self {
        Probe(1.0)
    }
    fn shape(&self) -> Shape {
        Shape::scalar()
    }
}

impl Elementary for Probe {
    fn exp(&self) -> Self {
        Probe(self.0.exp())
    }
    fn ln(&self) -> Self {
        Probe(self.0.ln())
    }
    fn sqrt(&self) -> Self {
        Probe(self.0.sqrt())
    }
    fn tanh(&self) -> Self {
        Probe(self.0.tanh())
    }
    fn powf(&self, exponent: Self) -> Self {
        Probe(self.0.powf(exponent.0))
    }
    fn maximum(&self, other: &Self) -> Self {
        Probe(self.0.max(other.0))
    }
    fn step(&self, threshold: &Self) -> Self {
        Probe(if self.0 >= threshold.0 { 1.0 } else { 0.0 })
    }
    fn gemm(task: &GemmTask<'_, Self>) -> Option<Vec<Self>> {
        Some(vec![Probe(42.0); task.m() * task.n()])
    }
}

#[test]
fn the_element_seam_answers_before_the_built_in_kernels() {
    let left = Tensor::new([2, 3], vec![Probe(1.0); 6]);
    let right = Tensor::new([3, 2], vec![Probe(1.0); 6]);
    let product = left.matmul(&right);
    assert_eq!(product.to_vec(), vec![Probe(42.0); 4]);
}

#[test]
fn nested_tensor_elements_multiply_on_the_slice_path() {
    let element = |value: f64| Tensor::new([2], [value, value + 1.0]);
    let left = Tensor::new(
        [2, 2],
        vec![element(1.0), element(2.0), element(3.0), element(4.0)],
    );
    let right = Tensor::new(
        [2, 2],
        vec![element(0.5), element(1.5), element(2.5), element(3.5)],
    );
    let product = left.matmul(&right);
    let expected = reference(&left, &right);
    assert_eq!(product.to_vec(), expected);
}
