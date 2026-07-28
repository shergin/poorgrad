use super::Elementary;

/// Matrix, reduction, transpose, and explicit broadcasting operations for
/// graph payloads.
///
/// This trait extends [`Elementary`] because forward and backward evaluation
/// must be able to execute every operation that can be recorded. For `f32` and
/// `f64`, [`Tensorial::matmul`] is multiplication and the remaining methods use
/// scalar identity semantics. `Tensor<Element>` provides the rank-aware
/// implementations.
///
/// Graph operations still validate their recorded [`Shape`](super::Shape).
/// Consequently, matrix multiplication and named-axis operations reject
/// scalar [`Value`](crate::Value) nodes even though direct trait calls on
/// scalar payloads are defined.
///
/// Broadcasting is never implicit. [`Tensorial::broadcast_like`] expands a
/// single-value payload to a reference shape, while
/// [`Tensorial::broadcast_along`] repeats a payload along one specified axis.
/// These operations are adjoint to [`Tensorial::sum`] and
/// [`Tensorial::sum_along`], respectively.
pub trait Tensorial: Elementary {
    /// Returns the matrix product of `self` and `rhs`.
    fn matmul(&self, rhs: &Self) -> Self;

    /// Returns `self` with its two axes swapped.
    fn transposed(&self) -> Self;

    /// Returns the sum of every value in `self`, shaped as a single value.
    fn sum(&self) -> Self;

    /// Returns `self` with `axis` reduced by summation: the result's
    /// shape is `self`'s with that axis removed.
    fn sum_along(&self, axis: usize) -> Self;

    /// Returns this payload's single value spread across `reference`'s
    /// shape.
    fn broadcast_like(&self, reference: &Self) -> Self;

    /// Returns `self` repeated along `axis` to match `reference`'s
    /// shape; `self`'s shape must equal `reference`'s with that axis
    /// removed.
    fn broadcast_along(&self, axis: usize, reference: &Self) -> Self;
}

impl Tensorial for f32 {
    fn matmul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn transposed(&self) -> Self {
        *self
    }

    fn sum(&self) -> Self {
        *self
    }

    fn sum_along(&self, _axis: usize) -> Self {
        *self
    }

    fn broadcast_like(&self, _reference: &Self) -> Self {
        *self
    }

    fn broadcast_along(&self, _axis: usize, _reference: &Self) -> Self {
        *self
    }
}

impl Tensorial for f64 {
    fn matmul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn transposed(&self) -> Self {
        *self
    }

    fn sum(&self) -> Self {
        *self
    }

    fn sum_along(&self, _axis: usize) -> Self {
        *self
    }

    fn broadcast_like(&self, _reference: &Self) -> Self {
        *self
    }

    fn broadcast_along(&self, _axis: usize, _reference: &Self) -> Self {
        *self
    }
}
