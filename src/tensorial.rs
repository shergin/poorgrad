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
/// bound.
///
/// Broadcasting is explicit by design: `broadcast_like` is the only way a
/// payload changes shape, it accepts only a single-value payload, and
/// every other operation demands exact shape agreement. There are no
/// implicit alignment rules.
pub trait Tensorial: Elementary {
    /// Returns the matrix product of `self` and `rhs`.
    fn matmul(&self, rhs: &Self) -> Self;

    /// Returns `self` with its two axes swapped.
    fn transposed(&self) -> Self;

    /// Returns the sum of every value in `self`, shaped as a single value.
    fn sum(&self) -> Self;

    /// Returns this payload's single value spread across `reference`'s
    /// shape.
    fn broadcast_like(&self, reference: &Self) -> Self;
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

    fn broadcast_like(&self, _reference: &Self) -> Self {
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

    fn broadcast_like(&self, _reference: &Self) -> Self {
        *self
    }
}
