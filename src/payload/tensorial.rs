use super::{Elementary, Shape};

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
    fn transpose(&self) -> Self;

    /// Returns the sum of every value in `self`, shaped as a single value.
    fn sum(&self) -> Self;

    /// Returns `self` with `axis` reduced by summation: the result's
    /// shape is `self`'s with that axis removed.
    fn sum_along(&self, axis: usize) -> Self;

    /// Returns `self` with `axis` reduced to its largest value by the
    /// elementwise [`maximum`](Elementary::maximum): the result's shape is
    /// `self`'s with that axis removed.
    ///
    /// It is the reduction behind stable normalization (`log_softmax`
    /// shifts by the axis maximum before exponentiating) and is not a
    /// recorded graph operation of its own.
    fn max_along(&self, axis: usize) -> Self;

    /// Returns this payload's single value spread across `reference`'s
    /// shape.
    fn broadcast_like(&self, reference: &Self) -> Self;

    /// Returns `self` repeated along `axis` to match `reference`'s
    /// shape; `self`'s shape must equal `reference`'s with that axis
    /// removed.
    fn broadcast_along(&self, axis: usize, reference: &Self) -> Self;

    /// Returns `self` reinterpreted with `shape`, preserving logical
    /// row-major order; the volume must not change.
    fn reshape(&self, shape: Shape) -> Self;

    /// Returns `self` with its axes reordered so that axis `i` of the
    /// result takes axis `order[i]` of `self`; `order` must be a
    /// permutation of `0..rank`.
    fn permute(&self, order: &[usize]) -> Self;

    /// Returns the window of `len` elements from `start` along `axis`:
    /// `self` with that axis restricted to `start .. start + len`. The
    /// window must hold at least one element, because tensors are never
    /// empty.
    fn narrow(&self, axis: usize, start: usize, len: usize) -> Self;

    /// Returns `self` placed into a zero payload whose `axis` has extent
    /// `full_extent`, at `start ..`, with zeros elsewhere: the adjoint of
    /// [`narrow`](Tensorial::narrow) and the gradient rule for it.
    fn pad(&self, axis: usize, start: usize, full_extent: usize) -> Self;

    /// Returns the sliding windows of `self` along `axis`: the axis is
    /// replaced by a `(count, size)` pair where window `w` starts at
    /// `w * step` and takes every `dilation`-th element, so
    /// `count = (extent - dilation * (size - 1) - 1) / step + 1`.
    ///
    /// It is the windowing view behind convolution and pooling (the
    /// torch-semantics single-axis `unfold`; two applications produce 2-D
    /// windows). Windows overlap when `step < dilation * size`, which is
    /// safe read-only aliasing: payloads are immutable.
    fn unfold(&self, axis: usize, size: usize, step: usize, dilation: usize) -> Self;

    /// Returns the `(count, size)` window pair at `axis`, `axis + 1`
    /// folded back onto an axis of `extent`: the adjoint of
    /// [`unfold`](Tensorial::unfold) and the gradient rule for it.
    ///
    /// Each source position sums the window elements that were read from
    /// it, accumulated output-centrically in window order, so the result
    /// is deterministic under any evaluation strategy. Positions no
    /// window reaches fold to zero.
    fn fold(&self, axis: usize, size: usize, step: usize, dilation: usize, extent: usize) -> Self;

    /// Returns the rows of `self` selected by `selection` (a one-hot
    /// `[count, vocab]` whose vocabulary matches `self`'s first axis): the
    /// embedding-style row gather, `result[i] = self[selection_index(i)]`.
    fn gather(&self, selection: &Self) -> Self;

    /// Scatter-adds the rows of `self` into a zero payload of `rows` rows by
    /// `selection`'s indices: the adjoint of [`gather`](Tensorial::gather)
    /// and its gradient rule, accumulating rows selected more than once.
    fn scatter(&self, selection: &Self, rows: usize) -> Self;
}

impl Tensorial for f32 {
    fn matmul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn transpose(&self) -> Self {
        *self
    }

    fn sum(&self) -> Self {
        *self
    }

    fn sum_along(&self, _axis: usize) -> Self {
        *self
    }

    fn max_along(&self, _axis: usize) -> Self {
        *self
    }

    fn broadcast_like(&self, _reference: &Self) -> Self {
        *self
    }

    fn broadcast_along(&self, _axis: usize, _reference: &Self) -> Self {
        *self
    }

    fn reshape(&self, _shape: Shape) -> Self {
        *self
    }

    fn permute(&self, _order: &[usize]) -> Self {
        *self
    }

    fn narrow(&self, _axis: usize, _start: usize, _len: usize) -> Self {
        *self
    }

    fn pad(&self, _axis: usize, _start: usize, _full_extent: usize) -> Self {
        *self
    }

    fn unfold(&self, _axis: usize, _size: usize, _step: usize, _dilation: usize) -> Self {
        *self
    }

    fn fold(
        &self,
        _axis: usize,
        _size: usize,
        _step: usize,
        _dilation: usize,
        _extent: usize,
    ) -> Self {
        *self
    }

    fn gather(&self, _selection: &Self) -> Self {
        *self
    }

    fn scatter(&self, _selection: &Self, _rows: usize) -> Self {
        *self
    }
}

impl Tensorial for f64 {
    fn matmul(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn transpose(&self) -> Self {
        *self
    }

    fn sum(&self) -> Self {
        *self
    }

    fn sum_along(&self, _axis: usize) -> Self {
        *self
    }

    fn max_along(&self, _axis: usize) -> Self {
        *self
    }

    fn broadcast_like(&self, _reference: &Self) -> Self {
        *self
    }

    fn broadcast_along(&self, _axis: usize, _reference: &Self) -> Self {
        *self
    }

    fn reshape(&self, _shape: Shape) -> Self {
        *self
    }

    fn permute(&self, _order: &[usize]) -> Self {
        *self
    }

    fn narrow(&self, _axis: usize, _start: usize, _len: usize) -> Self {
        *self
    }

    fn pad(&self, _axis: usize, _start: usize, _full_extent: usize) -> Self {
        *self
    }

    fn unfold(&self, _axis: usize, _size: usize, _step: usize, _dilation: usize) -> Self {
        *self
    }

    fn fold(
        &self,
        _axis: usize,
        _size: usize,
        _step: usize,
        _dilation: usize,
        _extent: usize,
    ) -> Self {
        *self
    }

    fn gather(&self, _selection: &Self) -> Self {
        *self
    }

    fn scatter(&self, _selection: &Self, _rows: usize) -> Self {
        *self
    }
}
