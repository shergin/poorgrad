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
    pub(crate) fn transpose(&self) -> Self {
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

    /// Returns the width of the buffer window this layout addresses: one
    /// past the distance from its first to its last logical element.
    ///
    /// A whole-window walk visits `span()` buffer elements where a logical
    /// walk visits `volume()`, so a caller trades one for the other by
    /// comparing the two; a broadcast view's span never exceeds its volume.
    pub(crate) fn span(&self) -> usize {
        let furthest: usize = self
            .shape
            .axes()
            .iter()
            .zip(&self.strides)
            .map(|(&extent, &stride)| (extent - 1) * stride)
            .sum();
        furthest + 1
    }

    /// Returns the layout with its offset rebased to zero, for addressing
    /// a fresh buffer that holds exactly the addressed window.
    pub(crate) fn rebased(&self) -> Self {
        Self {
            shape: self.shape.clone(),
            strides: self.strides.clone(),
            offset: 0,
        }
    }

    /// Returns the extent of the innermost axis: the length of one
    /// logical run. A rank-0 layout answers 1, its whole volume.
    pub(crate) fn inner_extent(&self) -> usize {
        self.shape.axes().last().copied().unwrap_or(1)
    }

    /// Returns the stride of the innermost axis: how a logical run steps
    /// through the buffer. A rank-0 layout answers 1, the unit step its
    /// single-element run never takes.
    pub(crate) fn inner_stride(&self) -> usize {
        self.strides.last().copied().unwrap_or(1)
    }

    /// Returns an iterator over the storage offsets at which each
    /// innermost-axis run begins, in logical order.
    pub(crate) fn run_offsets(&self) -> RunOffsets<'_> {
        let outer_rank = self.rank().saturating_sub(1);
        RunOffsets {
            layout: self,
            coordinates: std::iter::repeat_n(0, outer_rank).collect(),
            offset: self.offset,
            remaining: self.shape.axes()[..outer_rank].iter().product(),
        }
    }

    /// Returns a layout for `shape` over the same buffer region, preserving
    /// the offset, or `None` when the reshape must copy.
    ///
    /// A contiguous layout reaches any shape with freshly computed row-major
    /// strides. A strided layout survives only a reshape that inserts or
    /// removes extent-1 axes -- a squeeze or unsqueeze -- since such axes
    /// never advance the buffer and the remaining axes keep their strides.
    ///
    /// The caller guarantees `shape` has the same volume.
    pub(crate) fn reshape(&self, shape: Shape) -> Option<Layout> {
        if self.is_contiguous() {
            let strides = Self::contiguous_strides(&shape);
            return Some(Layout {
                shape,
                strides,
                offset: self.offset,
            });
        }
        self.unit_axis_view(shape)
    }

    /// Returns the view of a reshape that only inserts or removes extent-1
    /// axes, or `None` when the non-unit extents differ in sequence.
    ///
    /// Each non-unit target axis takes the stride of its matching non-unit
    /// source axis, and a unit axis takes stride 0, which no index ever
    /// applies.
    fn unit_axis_view(&self, shape: Shape) -> Option<Layout> {
        let mut source = self
            .shape
            .axes()
            .iter()
            .zip(&self.strides)
            .filter(|&(&extent, _)| extent != 1);
        let mut strides = Strides::new();
        for &extent in shape.axes() {
            if extent == 1 {
                strides.push(0);
                continue;
            }
            let (&matched, &stride) = source.next()?;
            if matched != extent {
                return None;
            }
            strides.push(stride);
        }
        if source.next().is_some() {
            return None;
        }
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
    pub(crate) fn permute(&self, order: &[usize]) -> Layout {
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
    pub(crate) fn narrow(&self, axis: usize, start: usize, len: usize) -> Layout {
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

    /// Returns the layout of sliding windows along `axis`, a view sharing
    /// the buffer: the axis is replaced by a `(count, size)` pair whose
    /// window-start stride is `step` steps and whose in-window stride is
    /// `dilation` steps of the original axis stride. Overlapping windows
    /// (`step < dilation * size`) alias buffer elements, which is safe
    /// because no operation ever writes through a view.
    ///
    /// The caller guarantees nonzero `size`, `step`, and `dilation`, and
    /// `dilation * (size - 1) + 1 <= extent(axis)`.
    pub(crate) fn unfold(&self, axis: usize, size: usize, step: usize, dilation: usize) -> Layout {
        let axes = self.shape.axes();
        let count = (axes[axis] - dilation * (size - 1) - 1) / step + 1;
        let mut unfolded: Vec<usize> = axes.to_vec();
        unfolded[axis] = count;
        unfolded.insert(axis + 1, size);
        let mut strides = self.strides.clone();
        let along = strides[axis];
        strides[axis] = step * along;
        strides.insert(axis + 1, dilation * along);
        Layout {
            shape: Shape::new(unfolded),
            strides,
            offset: self.offset,
        }
    }
}

/// Iterator over the storage offsets at which each innermost-axis run of a
/// [`Layout`] begins, in logical order: an odometer over the outer axes,
/// mirroring the element iterator's walk one run at a time.
pub(crate) struct RunOffsets<'layout> {
    layout: &'layout Layout,
    coordinates: Strides,
    offset: usize,
    remaining: usize,
}

impl Iterator for RunOffsets<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            return None;
        }
        let start = self.offset;
        self.remaining -= 1;
        if self.remaining > 0 {
            let axes = self.layout.shape.axes();
            // Advance the odometer: step the innermost outer axis, carrying
            // into the axes above it and adjusting the offset by the stride
            // of whichever axis moved.
            for axis in (0..self.coordinates.len()).rev() {
                self.coordinates[axis] += 1;
                if self.coordinates[axis] < axes[axis] {
                    self.offset += self.layout.strides[axis];
                    break;
                }
                self.offset -= (axes[axis] - 1) * self.layout.strides[axis];
                self.coordinates[axis] = 0;
            }
        }
        Some(start)
    }
}

#[cfg(test)]
#[path = "tests/layout_tests.rs"]
mod tests;
