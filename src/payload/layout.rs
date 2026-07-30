use smallvec::SmallVec;

use super::Shape;

/// The strides of a payload: how many flat-buffer elements to advance for a
/// unit step along each axis, parallel to a [`Shape`].
///
/// A stride of `0` marks a broadcast axis, whose steps do not move within
/// the buffer. Strides are stored inline through rank 4, mirroring `Shape`.
pub(crate) type Strides = SmallVec<[usize; 4]>;

/// How a [`Tensor`](super::Tensor)'s logical indices map onto its flat
/// storage: the shape, the per-axis strides, and the offset of the first
/// element.
///
/// The element at multi-index `(i0, ..., in)` lives at
/// `offset + sum(i_k * strides_k)` in the flat buffer. A row-major
/// contiguous layout has `strides_k = product(shape[k + 1 ..])` and
/// `offset = 0`. View operations (transpose, broadcast, and later reshape
/// and slice) produce a new layout over a shared buffer without moving any
/// element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Layout {
    shape: Shape,
    strides: Strides,
    offset: usize,
}

impl Layout {
    /// Creates the row-major contiguous layout of `shape`, starting at
    /// offset zero.
    pub(crate) fn contiguous(shape: Shape) -> Self {
        let strides = Self::contiguous_strides(&shape);
        Self {
            shape,
            strides,
            offset: 0,
        }
    }

    /// Returns the row-major strides of `shape`: each axis strides by the
    /// product of the extents to its right.
    fn contiguous_strides(shape: &Shape) -> Strides {
        let axes = shape.axes();
        let mut strides: Strides = std::iter::repeat_n(0usize, axes.len()).collect();
        let mut running = 1;
        for axis in (0..axes.len()).rev() {
            strides[axis] = running;
            running *= axes[axis];
        }
        strides
    }

    /// Returns the shape this layout addresses.
    pub(crate) fn shape(&self) -> &Shape {
        &self.shape
    }

    /// Returns the per-axis strides, parallel to the shape.
    pub(crate) fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Returns the flat-buffer index of the first logical element.
    pub(crate) fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the number of axes.
    pub(crate) fn rank(&self) -> usize {
        self.shape.rank()
    }

    /// Returns the number of logical elements.
    pub(crate) fn volume(&self) -> usize {
        self.shape.volume()
    }

    /// Returns whether the layout addresses a contiguous row-major slice of
    /// the buffer starting at its offset.
    ///
    /// Extent-1 axes impose no constraint, since their stride is never used,
    /// while a stride-0 broadcast axis of extent above one is never
    /// contiguous.
    pub(crate) fn is_contiguous(&self) -> bool {
        let axes = self.shape.axes();
        let mut expected = 1;
        for axis in (0..axes.len()).rev() {
            let extent = axes[axis];
            if extent != 1 && self.strides[axis] != expected {
                return false;
            }
            expected *= extent;
        }
        true
    }

    /// Returns the flat-buffer index of the logical row-major `position`.
    ///
    /// It unravels `position` into a multi-index and applies the strides and
    /// offset. This is the general per-element address; a contiguous layout
    /// has a faster slice path.
    pub(crate) fn storage_index(&self, position: usize) -> usize {
        let axes = self.shape.axes();
        let mut remainder = position;
        let mut index = self.offset;
        for axis in (0..axes.len()).rev() {
            let extent = axes[axis];
            index += (remainder % extent) * self.strides[axis];
            remainder /= extent;
        }
        index
    }

    /// Returns the layout with its two axes swapped.
    ///
    /// Rank 0 and rank 1 are returned unchanged.
    ///
    /// # Panics
    /// Panics if the rank exceeds 2.
    pub(crate) fn transposed(&self) -> Self {
        if self.rank() < 2 {
            return self.clone();
        }
        assert_eq!(self.rank(), 2, "transpose supports rank 2 at most");
        let axes = self.shape.axes();
        Self {
            shape: Shape::new([axes[1], axes[0]]),
            strides: [self.strides[1], self.strides[0]].into_iter().collect(),
            offset: self.offset,
        }
    }

    /// Returns the layout of this payload repeated along `axis` to fill
    /// `reference`: the current strides with a stride-0 axis inserted at
    /// `axis`.
    ///
    /// The caller guarantees that `self`'s shape equals `reference` with
    /// `axis` removed.
    pub(crate) fn broadcast_along(&self, axis: usize, reference: &Shape) -> Self {
        let mut strides = self.strides.clone();
        strides.insert(axis, 0);
        Self {
            shape: reference.clone(),
            strides,
            offset: self.offset,
        }
    }

    /// Returns a contiguous layout for `shape` over the same buffer region,
    /// preserving the offset, or `None` when this layout is not contiguous
    /// and the reshape must therefore copy.
    ///
    /// The caller guarantees `shape` has the same volume.
    pub(crate) fn reshaped(&self, shape: Shape) -> Option<Layout> {
        if !self.is_contiguous() {
            return None;
        }
        let strides = Self::contiguous_strides(&shape);
        Some(Layout {
            shape,
            strides,
            offset: self.offset,
        })
    }

    /// Returns the layout with its axes reordered by `order`: axis `i` of
    /// the result takes axis `order[i]` of `self`.
    ///
    /// The caller guarantees `order` is a permutation of `0..rank`.
    pub(crate) fn permuted(&self, order: &[usize]) -> Layout {
        let axes = self.shape.axes();
        Layout {
            shape: Shape::new(order.iter().map(|&axis| axes[axis])),
            strides: order.iter().map(|&axis| self.strides[axis]).collect(),
            offset: self.offset,
        }
    }

    /// Returns the layout of a window of `len` elements starting at `start`
    /// along `axis`, a view sharing the buffer: the offset advances by
    /// `start` steps of that axis's stride and the axis extent shrinks to
    /// `len`.
    ///
    /// The caller guarantees `start + len <= extent(axis)`.
    pub(crate) fn narrowed(&self, axis: usize, start: usize, len: usize) -> Layout {
        Layout {
            shape: Shape::new(
                self.shape
                    .axes()
                    .iter()
                    .enumerate()
                    .map(|(index, &extent)| if index == axis { len } else { extent }),
            ),
            strides: self.strides.clone(),
            offset: self.offset + start * self.strides[axis],
        }
    }
}

#[cfg(test)]
#[path = "tests/layout_tests.rs"]
mod tests;
