use super::Elementary;

/// The tensor-native operations: matrix multiplication, reduction, and
/// explicit shape manipulation.
///
/// It extends `Elementary` the same way `Elementary` extends
/// `Differentiable`: running a graph requires the full tier, while
/// building and updating stay arithmetic-only. Scalars implement it
/// degenerately — a scalar is a rank-0 tensor, so `matmul` collapses to
/// multiplication and `transposed`, `sum`, and `broadcast_like` to the
/// identity — which is what keeps scalar graphs running under the same
/// bound. The degenerate impls exist for exactly that: satisfying the
/// bound of running a graph, not recording tensor-native expressions on
/// scalar networks — record-time shape inference demands proper ranks
/// (`matmul` requires rank 2), so `Value::matmul` on a scalar network
/// panics at the offending expression.
///
/// Broadcasting is explicit by design, in two named forms: a
/// single-value payload spread across a reference's whole shape
/// (`broadcast_like`), or a payload repeated along one named axis of a
/// reference (`broadcast_along`). Every other operation demands exact
/// shape agreement; there are no implicit alignment rules. Each
/// broadcast form is adjoint to the matching reduction: `sum` to
/// `broadcast_like`, `sum_along` to `broadcast_along`.
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
