use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use static_assertions::assert_impl_all;

use super::elementary::MapOperation;
use super::gemm;
use super::layout::{Layout, Strides};
use super::storage::Storage;
use super::tensorial::composed_windowed_patches;
use super::{Differentiable, Elementary, Shape, Tensorial};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Tensor<f64>: Send, Sync);

/// A dense tensor with an immutable, runtime-defined [`Shape`] and a shared
/// element buffer read through a strided layout.
///
/// The elements are held behind a `Storage` representation: an
/// `Arc`-shared row-major buffer addressed by strides and an offset, or a
/// non-allocating constant. Cloning shares the buffer and clones only the
/// metadata; it does not clone the elements. Because tensors are immutable
/// and buffer-shared, view operations that alias a buffer (transpose and
/// broadcast) are always safe: no operation ever writes through an alias.
///
/// Arithmetic and [`Elementary`] functions operate elementwise in logical
/// row-major order. Binary elementwise operations require identical shapes
/// and never broadcast implicitly. Broadcasting is available only through
/// [`Tensorial::broadcast_like`] and [`Tensorial::broadcast_along`], and it
/// produces a view rather than copying.
///
/// [`Tensorial::matmul`] requires rank-2 operands, and
/// [`Tensorial::transpose`] accepts ranks 0 through 2, returning a view.
/// Reductions, explicit broadcasts, and reshaping are rank-general;
/// batched matrix multiplication is not supported.
#[derive(Debug, Clone)]
pub struct Tensor<Element> {
    storage: Storage<Element>,
}

impl<Element> Tensor<Element> {
    /// Returns the logical shape, the one descriptor every representation
    /// answers for.
    fn logical_shape(&self) -> &Shape {
        match &self.storage {
            Storage::Dense { layout, .. } => layout.shape(),
            Storage::Constant { shape, .. } => shape,
            Storage::Selection { shape, .. } => shape,
        }
    }

    /// Returns the row indices of a `Selection` payload.
    ///
    /// # Panics
    /// Panics if `self` is not a selection built with [`Tensor::selection`].
    fn selection_indices(&self) -> &[usize] {
        match &self.storage {
            Storage::Selection { indices, .. } => indices,
            _ => panic!("gather requires a selection tensor built with `Tensor::selection`"),
        }
    }

    /// Returns this payload as a gemm operand when it is a dense
    /// rank-2 tensor: the backing slice from the layout's offset,
    /// with the layout's strides. Other storages answer `None` and
    /// take the logical path.
    fn gemm_operand(&self) -> Option<(&[Element], [usize; 2])> {
        match &self.storage {
            Storage::Dense { data, layout } if layout.rank() == 2 => {
                let strides = [layout.strides()[0], layout.strides()[1]];
                Some((&data.as_slice()[layout.offset()..], strides))
            }
            _ => None,
        }
    }

    /// Returns the elements as a contiguous slice when the tensor is stored
    /// as a contiguous dense buffer, or `None` for a strided view or a
    /// constant.
    pub fn as_slice(&self) -> Option<&[Element]> {
        match &self.storage {
            Storage::Dense { data, layout } if layout.is_contiguous() => {
                let start = layout.offset();
                Some(&data.as_slice()[start..start + layout.volume()])
            }
            _ => None,
        }
    }

    /// Returns the buffer window a strided dense view addresses together
    /// with its layout rebased to the window start, or `None` when the
    /// storage is not a strided view or the window is not strictly narrower
    /// than the volume: only a broadcast view computes fewer elements and
    /// earns staying a view. An equal-width window (a transpose) must keep
    /// the logical scalar walk — the backend seam reading the window would
    /// silently replace the documented bitwise fallback — and a wider one
    /// (a narrow sliver) would cost more than the walk it replaces.
    ///
    /// An elementwise transform commutes with any view, so a caller may
    /// transform the window and keep the layout instead of materializing
    /// the logical order. The window can include buffer elements the view
    /// skips; the transform visits them, and no logical access reads them.
    fn strided_window(&self) -> Option<(&[Element], Layout)> {
        let Storage::Dense { data, layout } = &self.storage else {
            return None;
        };
        if layout.is_contiguous() || layout.span() >= layout.volume() {
            return None;
        }
        let start = layout.offset();
        Some((
            &data.as_slice()[start..start + layout.span()],
            layout.rebased(),
        ))
    }
}

impl<Element: Clone> Tensor<Element> {
    /// Returns the elements in logical row-major order.
    ///
    /// A contiguous dense buffer iterates its slice directly; a strided
    /// view walks its layout with an odometer; a constant repeats its
    /// single value across the shape's volume.
    ///
    /// It yields owned elements rather than references because a
    /// representation that computes its elements has nothing to lend,
    /// and for the numeric payloads a clone is the same load a
    /// borrow-then-copy compiles to.
    pub fn iter(&self) -> impl Iterator<Item = Element> + '_ {
        match &self.storage {
            Storage::Constant { shape, value } => ElementIter::Constant {
                value,
                remaining: shape.volume(),
            },
            Storage::Dense { data, layout } if layout.is_contiguous() => {
                let start = layout.offset();
                ElementIter::Contiguous(data.as_slice()[start..start + layout.volume()].iter())
            }
            Storage::Dense { data, layout } => ElementIter::Strided {
                data: data.as_slice(),
                shape: layout.shape().axes(),
                strides: layout.strides(),
                coordinates: std::iter::repeat_n(0usize, layout.rank()).collect(),
                index: layout.offset(),
                remaining: layout.volume(),
            },
            Storage::Selection {
                indices,
                shape,
                zero,
                one,
            } => ElementIter::Selection {
                indices: indices.as_slice(),
                vocab: shape.axes()[1],
                zero,
                one,
                position: 0,
                total: shape.volume(),
            },
        }
    }

    /// Returns the elements in logical row-major order as an owned vector.
    pub fn to_vec(&self) -> Vec<Element> {
        self.iter().collect()
    }

    /// Returns the element at logical row-major `position`.
    ///
    /// It is the general per-element read shared by every operation; a
    /// dense layout resolves it through the stride and offset arithmetic,
    /// and a constant answers with its single value.
    ///
    /// It returns an owned element rather than a reference because a
    /// representation that computes its elements has nothing to lend,
    /// and because nearly every caller wants the value: for the numeric
    /// payloads a clone is the same load a borrow-then-copy compiles to.
    fn get(&self, position: usize) -> Element {
        match &self.storage {
            Storage::Dense { data, layout } => data[layout.storage_index(position)].clone(),
            Storage::Constant { value, .. } => value.clone(),
            Storage::Selection {
                indices,
                shape,
                zero,
                one,
            } => {
                let vocab = shape.axes()[1];
                if indices[position / vocab] == position % vocab {
                    one.clone()
                } else {
                    zero.clone()
                }
            }
        }
    }
}

impl<Element: Differentiable> Tensor<Element> {
    /// Creates a tensor of `shape` from `elements` in row-major order.
    ///
    /// # Panics
    /// Panics if the shape's volume overflows `usize`, the number of elements
    /// differs from that volume, or the shape holds no elements. Empty tensors
    /// are unsupported because reductions initialize their accumulator from
    /// an existing element; [`Differentiable`] provides shape-preserving
    /// identities rather than a nullary element constructor.
    pub fn new(shape: impl Into<Shape>, elements: impl Into<Vec<Element>>) -> Self {
        Self::dense(shape.into(), elements.into())
    }

    /// Creates a tensor of `shape` with every element set to `element`,
    /// stored as a non-allocating constant.
    ///
    /// # Panics
    /// Panics if the shape's volume overflows `usize` or the shape holds no
    /// elements, as documented on [`Tensor::new`].
    pub fn filled(shape: impl Into<Shape>, element: Element) -> Self {
        Self::constant(shape.into(), element)
    }

    /// Builds a contiguous dense tensor of `shape` from `elements`.
    ///
    /// Every dense tensor is created here, so this is where the tensor
    /// invariant is proven: the shape's volume is representable (checked
    /// by [`Shape::volume`]), positive, and equal to the buffer length.
    fn dense(shape: Shape, elements: Vec<Element>) -> Self {
        assert_eq!(
            shape.volume(),
            elements.len(),
            "tensor shape does not match its number of elements"
        );
        assert!(
            !elements.is_empty(),
            "tensors must hold at least one element"
        );
        Self {
            storage: Storage::Dense {
                layout: Layout::contiguous(shape),
                data: Arc::new(elements),
            },
        }
    }

    /// Builds a constant tensor of `shape` filled with `value`.
    ///
    /// Every constant tensor is created here, so the tensor invariant is
    /// proven here as on [`Tensor::dense`]: a representable, positive
    /// volume.
    fn constant(shape: Shape, value: Element) -> Self {
        assert!(shape.volume() > 0, "tensors must hold at least one element");
        Self {
            storage: Storage::Constant { shape, value },
        }
    }

    /// Creates the one-hot `[indices.len(), vocab]` selection matrix whose
    /// row `i` is `one` at column `indices[i]` and zero elsewhere, stored as
    /// its indices rather than a dense buffer.
    ///
    /// It carries the token indices of an embedding lookup: feed it as a
    /// per-run input and read it with [`Tensorial::gather`](super::Tensorial::gather).
    /// `one` is the value placed at each selected position (the
    /// multiplicative identity, e.g. `1.0`); the zero is derived from it.
    ///
    /// # Panics
    /// Panics if `vocab` is zero, `indices` is empty, any index is not
    /// below `vocab`, or the `[indices.len(), vocab]` volume overflows
    /// `usize`.
    pub fn selection(indices: impl Into<Vec<usize>>, vocab: usize, one: Element) -> Self {
        let indices = indices.into();
        assert!(vocab > 0, "a selection needs a non-empty vocabulary");
        assert!(
            !indices.is_empty(),
            "tensors must hold at least one element"
        );
        assert!(
            indices.len().checked_mul(vocab).is_some(),
            "shape volume overflows `usize`"
        );
        for &index in &indices {
            assert!(
                index < vocab,
                "selection index {index} is out of vocabulary {vocab}"
            );
        }
        let zero = one.zero_like();
        let shape = Shape::new([indices.len(), vocab]);
        Self {
            storage: Storage::Selection {
                indices: Arc::new(indices),
                shape,
                zero,
                one,
            },
        }
    }

    /// Returns an equivalent contiguous dense tensor, materializing any
    /// non-dense or strided representation.
    ///
    /// It is the correctness fallback for view operations that a `Selection`
    /// does not model directly (transpose, permute, narrow, axis broadcast):
    /// densify first, then take the dense view.
    fn densify(&self) -> Self {
        match &self.storage {
            Storage::Dense { layout, .. } if layout.is_contiguous() => self.clone(),
            _ => Self::dense(self.logical_shape().clone(), self.to_vec()),
        }
    }

    /// Returns a tensor with every element passed through `transform`.
    ///
    /// A constant maps in place to another constant; a contiguous
    /// dense buffer maps over its slice directly — bypassing the
    /// per-element iterator dispatch, which measured well below
    /// memory speed; a strided view maps its buffer window and keeps
    /// the layout, so a broadcast view transforms only its distinct
    /// elements and stays a view; and everything else materializes
    /// through the logical-order iterator. A pure `transform` gives
    /// every logical element the same value on every path, so the
    /// lanes agree bitwise.
    fn map(&self, transform: impl Fn(&Element) -> Element) -> Self {
        if let Storage::Constant { shape, value } = &self.storage {
            return Self::constant(shape.clone(), transform(value));
        }
        if let Some(elements) = self.as_slice() {
            return Self::dense(
                self.logical_shape().clone(),
                elements.iter().map(transform).collect(),
            );
        }
        if let Some((window, layout)) = self.strided_window() {
            return Self {
                storage: Storage::Dense {
                    data: Arc::new(window.iter().map(transform).collect()),
                    layout,
                },
            };
        }
        Self::dense(
            self.logical_shape().clone(),
            self.iter().map(|element| transform(&element)).collect(),
        )
    }

    /// Combines two tensors element by element with `combine`.
    ///
    /// Two constants combine into a constant in O(1); contiguous
    /// dense buffers combine slice to slice; a dense-and-constant
    /// pair maps the dense operand against the single value; two
    /// dense buffers whose innermost strides are unit or zero
    /// combine run by run as slices; and everything else goes
    /// through the logical-order iterators. A pure `combine` hands
    /// every logical pair the same value on every path, so the
    /// lanes agree bitwise.
    ///
    /// # Panics
    /// Panics if the tensors have different shapes.
    fn zip(&self, other: &Self, combine: impl Fn(&Element, &Element) -> Element) -> Self {
        assert_eq!(
            self.logical_shape(),
            other.logical_shape(),
            "tensors have different shapes"
        );
        if let (Storage::Constant { value: left, .. }, Storage::Constant { value: right, .. }) =
            (&self.storage, &other.storage)
        {
            return Self::constant(self.logical_shape().clone(), combine(left, right));
        }
        if let (Some(left), Some(right)) = (self.as_slice(), other.as_slice()) {
            return Self::dense(
                self.logical_shape().clone(),
                left.iter()
                    .zip(right)
                    .map(|(left, right)| combine(left, right))
                    .collect(),
            );
        }
        if let (Storage::Dense { .. }, Storage::Constant { value: right, .. }) =
            (&self.storage, &other.storage)
        {
            return self.map(|left| combine(left, right));
        }
        if let (Storage::Constant { value: left, .. }, Storage::Dense { .. }) =
            (&self.storage, &other.storage)
        {
            return other.map(|right| combine(left, right));
        }
        if let (
            Storage::Dense {
                data: left,
                layout: left_layout,
            },
            Storage::Dense {
                data: right,
                layout: right_layout,
            },
        ) = (&self.storage, &other.storage)
            && let Some(combined) = zipped_runs(left, left_layout, right, right_layout, &combine)
        {
            return Self::dense(self.logical_shape().clone(), combined);
        }
        Self::dense(
            self.logical_shape().clone(),
            self.iter()
                .zip(other.iter())
                .map(|(left, right)| combine(&left, &right))
                .collect(),
        )
    }
}

/// Combines two same-shape dense buffers by innermost-axis runs, or
/// declines with `None` when either innermost stride exceeds one.
///
/// A unit stride walks a run as a slice and a zero stride holds one
/// element for the whole run, so every accepted case combines slices or
/// a slice against a held element — the loop shapes vectorization needs,
/// where the logical-order odometer defeats it. Both operands advance
/// run by run through their outer-axis offsets in logical order; a
/// zero-by-zero run computes its one value and repeats it.
fn zipped_runs<Element: Clone>(
    left: &[Element],
    left_layout: &Layout,
    right: &[Element],
    right_layout: &Layout,
    combine: impl Fn(&Element, &Element) -> Element,
) -> Option<Vec<Element>> {
    let left_stride = left_layout.inner_stride();
    let right_stride = right_layout.inner_stride();
    if left_stride > 1 || right_stride > 1 {
        return None;
    }
    let extent = left_layout.inner_extent();
    let mut combined = Vec::with_capacity(left_layout.volume());
    for (left_start, right_start) in left_layout.run_offsets().zip(right_layout.run_offsets()) {
        match (left_stride, right_stride) {
            (1, 1) => combined.extend(
                left[left_start..left_start + extent]
                    .iter()
                    .zip(&right[right_start..right_start + extent])
                    .map(|(left, right)| combine(left, right)),
            ),
            (1, 0) => {
                let held = &right[right_start];
                combined.extend(
                    left[left_start..left_start + extent]
                        .iter()
                        .map(|left| combine(left, held)),
                );
            }
            (0, 1) => {
                let held = &left[left_start];
                combined.extend(
                    right[right_start..right_start + extent]
                        .iter()
                        .map(|right| combine(held, right)),
                );
            }
            _ => {
                let value = combine(&left[left_start], &right[right_start]);
                combined.extend(std::iter::repeat_n(value, extent));
            }
        }
    }
    Some(combined)
}

/// Returns the shape with its two axes swapped, matching the payload
/// transpose that a constant undergoes without touching a buffer.
///
/// # Panics
/// Panics if the rank exceeds 2.
fn transpose_shape(shape: &Shape) -> Shape {
    if shape.rank() < 2 {
        return shape.clone();
    }
    assert_eq!(shape.rank(), 2, "transpose supports rank 2 at most");
    let axes = shape.axes();
    Shape::new([axes[1], axes[0]])
}

/// Iterator over a tensor's elements in logical row-major order,
/// yielding owned clones so a representation without a buffer to
/// borrow from can still answer.
///
/// The variants mirror the storage representations: a repeated constant, a
/// direct slice walk for a contiguous buffer, and an odometer walk for a
/// strided view.
enum ElementIter<'tensor, Element> {
    Constant {
        value: &'tensor Element,
        remaining: usize,
    },
    Contiguous(std::slice::Iter<'tensor, Element>),
    Strided {
        data: &'tensor [Element],
        shape: &'tensor [usize],
        strides: &'tensor [usize],
        coordinates: Strides,
        index: usize,
        remaining: usize,
    },
    Selection {
        indices: &'tensor [usize],
        vocab: usize,
        zero: &'tensor Element,
        one: &'tensor Element,
        position: usize,
        total: usize,
    },
}

impl<'tensor, Element: Clone> Iterator for ElementIter<'tensor, Element> {
    type Item = Element;

    fn next(&mut self) -> Option<Element> {
        match self {
            ElementIter::Constant { value, remaining } => {
                if *remaining == 0 {
                    return None;
                }
                *remaining -= 1;
                Some((*value).clone())
            }
            ElementIter::Contiguous(iterator) => iterator.next().cloned(),
            ElementIter::Strided {
                data,
                shape,
                strides,
                coordinates,
                index,
                remaining,
            } => {
                if *remaining == 0 {
                    return None;
                }
                let element = data[*index].clone();
                *remaining -= 1;
                if *remaining > 0 {
                    // Advance the odometer: step the innermost axis, carrying
                    // into the outer axes and adjusting the flat index by the
                    // stride of whichever axis moved.
                    for axis in (0..shape.len()).rev() {
                        coordinates[axis] += 1;
                        if coordinates[axis] < shape[axis] {
                            *index += strides[axis];
                            break;
                        }
                        *index -= (shape[axis] - 1) * strides[axis];
                        coordinates[axis] = 0;
                    }
                }
                Some(element)
            }
            ElementIter::Selection {
                indices,
                vocab,
                zero,
                one,
                position,
                total,
            } => {
                if *position >= *total {
                    return None;
                }
                let row = *position / *vocab;
                let column = *position % *vocab;
                *position += 1;
                Some(if indices[row] == column {
                    (*one).clone()
                } else {
                    (*zero).clone()
                })
            }
        }
    }
}

impl<Element: PartialEq + Clone> PartialEq for Tensor<Element> {
    /// Compares two tensors by logical value: equal shapes and equal
    /// elements in logical order, independent of storage representation, so
    /// a view compares equal to its materialized twin.
    fn eq(&self, other: &Self) -> bool {
        self.logical_shape() == other.logical_shape() && self.iter().eq(other.iter())
    }
}

impl<Element: Differentiable> Add for Tensor<Element> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        self.zip(&rhs, |left, right| left.clone() + right.clone())
    }
}

impl<Element: Differentiable> Sub for Tensor<Element> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        self.zip(&rhs, |left, right| left.clone() - right.clone())
    }
}

impl<Element: Differentiable> Mul for Tensor<Element> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        self.zip(&rhs, |left, right| left.clone() * right.clone())
    }
}

impl<Element: Differentiable> Div for Tensor<Element> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        self.zip(&rhs, |left, right| left.clone() / right.clone())
    }
}

impl<Element: Differentiable> Neg for Tensor<Element> {
    type Output = Self;

    fn neg(self) -> Self {
        self.map(|element| -element.clone())
    }
}

impl<Element: Differentiable> Differentiable for Tensor<Element> {
    /// Returns a zero shaped like `self`, stored as a constant.
    ///
    /// It reads one element's `zero_like` as the fill, which is exact for
    /// the scalar payloads a tensor holds: their identity does not vary
    /// across the buffer.
    fn zero_like(&self) -> Self {
        Self::constant(self.logical_shape().clone(), self.get(0).zero_like())
    }

    /// Returns a one shaped like `self`, stored as a constant.
    fn one_like(&self) -> Self {
        Self::constant(self.logical_shape().clone(), self.get(0).one_like())
    }

    /// Returns the count spread across `shape`, stored as a constant;
    /// the element value comes from the element type's own `counted`.
    ///
    /// # Panics
    /// Panics if the shape's volume overflows `usize` or the shape holds
    /// no elements, as documented on [`Tensor::new`].
    fn counted(shape: Shape, count: usize) -> Self {
        Self::constant(shape, Element::counted(Shape::scalar(), count))
    }

    fn shape(&self) -> Shape {
        self.logical_shape().clone()
    }
}

impl<Element: Elementary> Tensor<Element> {
    /// Applies one elementwise transcendental: the backend seam first
    /// for a contiguous dense buffer or for a strided view's buffer
    /// window — which keeps the view, so a broadcast operand hands the
    /// backend only its distinct elements — and the scalar `fallback`
    /// everywhere else (constants, declined maps).
    fn mapped(&self, operation: MapOperation, fallback: impl Fn(&Element) -> Element) -> Self {
        if let Some(elements) = self.as_slice()
            && let Some(mapped) = Element::map(operation, elements)
        {
            assert_eq!(
                mapped.len(),
                elements.len(),
                "the `Elementary::map` contract requires one output per input element"
            );
            return Self::dense(self.logical_shape().clone(), mapped);
        }
        if let Some((window, layout)) = self.strided_window()
            && let Some(mapped) = Element::map(operation, window)
        {
            assert_eq!(
                mapped.len(),
                window.len(),
                "the `Elementary::map` contract requires one output per input element"
            );
            return Self {
                storage: Storage::Dense {
                    data: Arc::new(mapped),
                    layout,
                },
            };
        }
        self.map(fallback)
    }
}

impl<Element: Elementary> Elementary for Tensor<Element> {
    fn exp(&self) -> Self {
        self.mapped(MapOperation::Exp, |element| element.exp())
    }

    fn ln(&self) -> Self {
        self.mapped(MapOperation::Ln, |element| element.ln())
    }

    fn sqrt(&self) -> Self {
        self.mapped(MapOperation::Sqrt, |element| element.sqrt())
    }

    fn tanh(&self) -> Self {
        self.mapped(MapOperation::Tanh, |element| element.tanh())
    }

    fn powf(&self, exponent: Self) -> Self {
        self.zip(&exponent, |element, exponent| {
            element.powf(exponent.clone())
        })
    }

    fn maximum(&self, other: &Self) -> Self {
        self.zip(other, |element, other| element.maximum(other))
    }

    fn step(&self, threshold: &Self) -> Self {
        self.zip(threshold, |element, threshold| element.step(threshold))
    }
}

impl<Element: Elementary> Tensorial for Tensor<Element> {
    /// Returns the matrix product of two rank-2 tensors.
    ///
    /// Dense operands, including strided views (a transposed operand,
    /// most often), multiply on a slice path that reads their buffers
    /// through the layout strides directly; other storages read
    /// through logical access. Both paths accumulate every output
    /// element in the same order, so their results are bit-identical.
    ///
    /// # Panics
    /// Panics if either operand is not rank 2, the inner dimensions do not
    /// agree, or any dimension is empty.
    fn matmul(&self, rhs: &Self) -> Self {
        let left = self.logical_shape();
        let right = rhs.logical_shape();
        assert_eq!(left.rank(), 2, "matmul requires rank-2 tensors");
        assert_eq!(right.rank(), 2, "matmul requires rank-2 tensors");
        let (rows, inner) = (left.axes()[0], left.axes()[1]);
        let (rhs_inner, columns) = (right.axes()[0], right.axes()[1]);
        assert_eq!(inner, rhs_inner, "matmul inner dimensions do not agree");
        assert!(
            rows > 0 && inner > 0 && columns > 0,
            "matmul requires non-empty dimensions"
        );

        if let (Some((a, a_strides)), Some((b, b_strides))) =
            (self.gemm_operand(), rhs.gemm_operand())
        {
            let task = gemm::GemmTask::new(a, a_strides, b, b_strides, rows, inner, columns);
            // Tier one is the element's acceleration seam (the
            // compiled backend chain); tier two the built-in slice
            // path. A declined task costs one call answering `None`.
            let elements = match Element::gemm(&task) {
                Some(product) => {
                    assert_eq!(
                        product.len(),
                        rows * columns,
                        "the `Elementary::gemm` contract requires `rows * columns` elements"
                    );
                    product
                }
                None => gemm::multiply(&task),
            };
            return Self::dense(Shape::new([rows, columns]), elements);
        }

        let mut elements = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                let mut total = self.get(row * inner) * rhs.get(column);
                for step in 1..inner {
                    total = total + self.get(row * inner + step) * rhs.get(step * columns + column);
                }
                elements.push(total);
            }
        }
        Self::dense(Shape::new([rows, columns]), elements)
    }

    /// Returns the tensor with its two axes swapped as a view over the same
    /// buffer.
    ///
    /// Rank-0 and rank-1 tensors are returned unchanged.
    ///
    /// # Panics
    /// Panics if the tensor's rank exceeds 2.
    fn transpose(&self) -> Self {
        match &self.storage {
            Storage::Dense { data, layout } => Self {
                storage: Storage::Dense {
                    data: Arc::clone(data),
                    layout: layout.transpose(),
                },
            },
            Storage::Constant { shape, value } => {
                Self::constant(transpose_shape(shape), value.clone())
            }
            Storage::Selection { .. } => self.densify().transpose(),
        }
    }

    /// Returns the sum of every element as a rank-0 constant.
    ///
    /// Elements are accumulated in logical order from left to right without
    /// pairwise or compensated summation.
    fn sum(&self) -> Self {
        let mut elements = self.iter();
        let first = elements.next().expect("sum requires a non-empty tensor");
        let total = elements.fold(first, |total, element| total + element);
        Self::constant(Shape::scalar(), total)
    }

    /// Returns the tensor with `axis` reduced by summation.
    ///
    /// The reduction is rank-general: the elements are viewed as
    /// `[outer, axis, inner]` in logical order and summed over the middle
    /// extent.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    fn sum_along(&self, axis: usize) -> Self {
        let shape = self.logical_shape();
        let axes = shape.axes();
        assert!(axis < axes.len(), "axis {axis} is out of rank for {shape}");
        let outer: usize = axes[..axis].iter().product();
        let extent = axes[axis];
        let inner: usize = axes[axis + 1..].iter().product();

        let mut elements = Vec::with_capacity(outer * inner);
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let position = |step: usize| (outer_index * extent + step) * inner + inner_index;
                let mut total = self.get(position(0));
                for step in 1..extent {
                    total = total + self.get(position(step));
                }
                elements.push(total);
            }
        }
        Self::dense(shape.without_axis(axis), elements)
    }

    /// Returns the tensor with `axis` reduced to its largest element by the
    /// elementwise [`Elementary::maximum`].
    ///
    /// The reduction is rank-general and mirrors [`Tensorial::sum_along`]:
    /// the elements are viewed as `[outer, axis, inner]` in logical order
    /// and folded over the middle extent.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank.
    fn max_along(&self, axis: usize) -> Self {
        let shape = self.logical_shape();
        let axes = shape.axes();
        assert!(axis < axes.len(), "axis {axis} is out of rank for {shape}");
        let outer: usize = axes[..axis].iter().product();
        let extent = axes[axis];
        let inner: usize = axes[axis + 1..].iter().product();

        let mut elements = Vec::with_capacity(outer * inner);
        for outer_index in 0..outer {
            for inner_index in 0..inner {
                let position = |step: usize| (outer_index * extent + step) * inner + inner_index;
                let mut largest = self.get(position(0));
                for step in 1..extent {
                    largest = largest.maximum(&self.get(position(step)));
                }
                elements.push(largest);
            }
        }
        Self::dense(shape.without_axis(axis), elements)
    }

    /// Returns this tensor's single element spread across `reference`'s
    /// shape as a constant: the whole-shape form of explicit broadcasting.
    ///
    /// # Panics
    /// Panics if `self` holds more than one element.
    fn broadcast_like(&self, reference: &Self) -> Self {
        assert_eq!(
            self.logical_shape().volume(),
            1,
            "broadcast requires a single-element tensor"
        );
        Self::constant(reference.logical_shape().clone(), self.get(0))
    }

    /// Returns the tensor repeated along `axis` to match `reference`'s
    /// shape as a stride-0 view: the named-axis form of explicit
    /// broadcasting.
    ///
    /// # Panics
    /// Panics if `axis` is out of `reference`'s rank or `self`'s shape
    /// differs from `reference`'s with that axis removed.
    fn broadcast_along(&self, axis: usize, reference: &Self) -> Self {
        let reference_shape = reference.logical_shape();
        let axes = reference_shape.axes();
        assert!(
            axis < axes.len(),
            "axis {axis} is out of rank for {reference_shape}"
        );
        assert_eq!(
            self.logical_shape(),
            &reference_shape.without_axis(axis),
            "broadcast along axis {axis} of {reference_shape} requires the remaining shape"
        );
        match &self.storage {
            Storage::Constant { value, .. } => {
                Self::constant(reference_shape.clone(), value.clone())
            }
            Storage::Dense { data, layout } => Self {
                storage: Storage::Dense {
                    data: Arc::clone(data),
                    layout: layout.broadcast_along(axis, reference_shape),
                },
            },
            Storage::Selection { .. } => self.densify().broadcast_along(axis, reference),
        }
    }

    /// Returns `self` reinterpreted with `shape` in logical row-major
    /// order.
    ///
    /// A contiguous dense tensor and a constant reshape into an O(1) view
    /// over the same buffer, and so does a strided view when only extent-1
    /// axes are inserted or removed; any other strided reshape is first
    /// materialized.
    ///
    /// # Panics
    /// Panics if `shape`'s volume differs from `self`'s.
    fn reshape(&self, shape: Shape) -> Self {
        assert_eq!(
            self.logical_shape().volume(),
            shape.volume(),
            "reshape from {} to {shape} changes the number of elements",
            self.logical_shape()
        );
        match &self.storage {
            Storage::Constant { value, .. } => Self::constant(shape, value.clone()),
            Storage::Dense { data, layout } => match layout.reshape(shape.clone()) {
                Some(reshaped) => Self {
                    storage: Storage::Dense {
                        data: Arc::clone(data),
                        layout: reshaped,
                    },
                },
                None => Self::dense(shape, self.to_vec()),
            },
            Storage::Selection { .. } => Self::dense(shape, self.to_vec()),
        }
    }

    /// Returns `self` with its axes reordered by `order` as a view over the
    /// same buffer.
    ///
    /// # Panics
    /// Panics if `order` is not a permutation of `0..rank`.
    fn permute(&self, order: &[usize]) -> Self {
        let shape = self.logical_shape();
        assert_eq!(
            order.len(),
            shape.rank(),
            "permute order must cover every axis of {shape}"
        );
        let mut seen = vec![false; shape.rank()];
        for &axis in order {
            assert!(
                axis < shape.rank(),
                "permute axis {axis} is out of rank for {shape}"
            );
            assert!(
                !std::mem::replace(&mut seen[axis], true),
                "permute order repeats axis {axis}"
            );
        }
        match &self.storage {
            Storage::Constant { value, .. } => {
                let axes = shape.axes();
                let permuted = Shape::new(order.iter().map(|&axis| axes[axis]));
                Self::constant(permuted, value.clone())
            }
            Storage::Dense { data, layout } => Self {
                storage: Storage::Dense {
                    data: Arc::clone(data),
                    layout: layout.permute(order),
                },
            },
            Storage::Selection { .. } => self.densify().permute(order),
        }
    }

    /// Returns the window of `len` elements from `start` along `axis` as a
    /// view over the same buffer.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank, `len` is zero (tensors cannot be
    /// empty), or `start + len` overflows or exceeds the axis extent.
    fn narrow(&self, axis: usize, start: usize, len: usize) -> Self {
        let shape = self.logical_shape();
        assert!(
            axis < shape.rank(),
            "narrow axis {axis} is out of rank for {shape}"
        );
        assert!(len > 0, "narrow window must hold at least one element");
        let extent = shape.axes()[axis];
        let end = start
            .checked_add(len)
            .expect("narrow window end overflows `usize`");
        assert!(
            end <= extent,
            "narrow window {start}..{end} exceeds axis {axis} extent {extent}"
        );
        match &self.storage {
            Storage::Constant { value, .. } => {
                let narrowed = Shape::new(
                    shape
                        .axes()
                        .iter()
                        .enumerate()
                        .map(|(index, &e)| if index == axis { len } else { e }),
                );
                Self::constant(narrowed, value.clone())
            }
            Storage::Dense { data, layout } => Self {
                storage: Storage::Dense {
                    data: Arc::clone(data),
                    layout: layout.narrow(axis, start, len),
                },
            },
            Storage::Selection { .. } => self.densify().narrow(axis, start, len),
        }
    }

    /// Returns `self` placed at `start ..` along `axis` inside a tensor
    /// whose `axis` has extent `full_extent`, with zeros elsewhere.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank or the window overflows or exceeds
    /// `full_extent`.
    fn pad(&self, axis: usize, start: usize, full_extent: usize) -> Self {
        let shape = self.logical_shape();
        assert!(
            axis < shape.rank(),
            "pad axis {axis} is out of rank for {shape}"
        );
        let axes = shape.axes();
        let len = axes[axis];
        let end = start
            .checked_add(len)
            .expect("pad window end overflows `usize`");
        assert!(
            end <= full_extent,
            "pad window {start}..{end} exceeds the full extent {full_extent}"
        );
        let outer: usize = axes[..axis].iter().product();
        let inner: usize = axes[axis + 1..].iter().product();
        let zero = self.get(0).zero_like();

        let mut elements = Vec::with_capacity(outer * full_extent * inner);
        for outer_index in 0..outer {
            for position in 0..full_extent {
                for inner_index in 0..inner {
                    if position >= start && position < end {
                        let source = (outer_index * len + (position - start)) * inner + inner_index;
                        elements.push(self.get(source));
                    } else {
                        elements.push(zero.clone());
                    }
                }
            }
        }
        let padded = Shape::new(
            axes.iter()
                .enumerate()
                .map(|(index, &e)| if index == axis { full_extent } else { e }),
        );
        Self::dense(padded, elements)
    }

    /// Returns the sliding windows of `self` along `axis` as a strided
    /// view over the same buffer: the axis becomes a `(count, size)`
    /// pair where window `w` starts at `w * step` and takes every
    /// `dilation`-th element. Overlapping windows alias elements
    /// read-only, which immutability makes safe.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank, `size`, `step`, or `dilation`
    /// is zero, or the dilated window span `dilation * (size - 1) + 1`
    /// overflows or exceeds the axis extent.
    fn unfold(&self, axis: usize, size: usize, step: usize, dilation: usize) -> Self {
        let shape = self.logical_shape();
        assert!(
            axis < shape.rank(),
            "unfold axis {axis} is out of rank for {shape}"
        );
        assert!(size > 0, "unfold windows must hold at least one element");
        assert!(step > 0, "unfold step must be positive");
        assert!(dilation > 0, "unfold dilation must be positive");
        let extent = shape.axes()[axis];
        let span = dilation
            .checked_mul(size - 1)
            .and_then(|reach| reach.checked_add(1))
            .expect("unfold window span overflows `usize`");
        assert!(
            span <= extent,
            "unfold window span {span} exceeds axis {axis} extent {extent}"
        );
        match &self.storage {
            Storage::Constant { value, .. } => {
                let count = (extent - span) / step + 1;
                let mut unfolded: Vec<usize> = shape.axes().to_vec();
                unfolded[axis] = count;
                unfolded.insert(axis + 1, size);
                Self::constant(Shape::new(unfolded), value.clone())
            }
            Storage::Dense { data, layout } => Self {
                storage: Storage::Dense {
                    data: Arc::clone(data),
                    layout: layout.unfold(axis, size, step, dilation),
                },
            },
            Storage::Selection { .. } => self.densify().unfold(axis, size, step, dilation),
        }
    }

    /// Returns the `(count, size)` window pair at `axis`, `axis + 1`
    /// folded back onto an axis of `extent`: the adjoint of
    /// [`unfold`](Tensorial::unfold) and its gradient rule.
    ///
    /// The accumulation is output-centric: each source position sums,
    /// in window order, the window elements that were read from it, so
    /// the result is deterministic under any evaluation strategy.
    /// Positions no window reaches fold to zero.
    ///
    /// # Panics
    /// Panics if `axis + 1` is out of rank, `size`, `step`, or
    /// `dilation` is zero, the dilated window span overflows or exceeds
    /// `extent`, or the shape at `axis`, `axis + 1` disagrees with the
    /// windows `unfold` would produce for `extent`.
    fn fold(&self, axis: usize, size: usize, step: usize, dilation: usize, extent: usize) -> Self {
        let shape = self.logical_shape();
        let axes = shape.axes();
        assert!(
            axis + 1 < axes.len(),
            "fold window axes {axis}, {} are out of rank for {shape}",
            axis + 1
        );
        assert!(size > 0, "fold windows must hold at least one element");
        assert!(step > 0, "fold step must be positive");
        assert!(dilation > 0, "fold dilation must be positive");
        let span = dilation
            .checked_mul(size - 1)
            .and_then(|reach| reach.checked_add(1))
            .expect("fold window span overflows `usize`");
        assert!(
            span <= extent,
            "fold window span {span} exceeds the extent {extent}"
        );
        let count = (extent - span) / step + 1;
        assert_eq!(
            axes[axis], count,
            "fold window count {} disagrees with the {count} windows of extent {extent}",
            axes[axis]
        );
        assert_eq!(
            axes[axis + 1],
            size,
            "fold window size {} disagrees with {size}",
            axes[axis + 1]
        );
        let outer: usize = axes[..axis].iter().product();
        let inner: usize = axes[axis + 2..].iter().product();
        let zero = self.get(0).zero_like();

        let mut elements = Vec::with_capacity(outer * extent * inner);
        for outer_index in 0..outer {
            for position in 0..extent {
                for inner_index in 0..inner {
                    let mut total = zero.clone();
                    // Window `w` reads `position` as its element `k`
                    // exactly when `w * step + k * dilation == position`.
                    for k in 0..size {
                        let reach = k * dilation;
                        if reach > position {
                            break;
                        }
                        let rest = position - reach;
                        if !rest.is_multiple_of(step) {
                            continue;
                        }
                        let window = rest / step;
                        if window >= count {
                            continue;
                        }
                        let source =
                            ((outer_index * count + window) * size + k) * inner + inner_index;
                        total = total + self.get(source);
                    }
                    elements.push(total);
                }
            }
        }
        let folded = Shape::new(
            axes[..axis]
                .iter()
                .copied()
                .chain(std::iter::once(extent))
                .chain(axes[axis + 2..].iter().copied()),
        );
        Self::dense(folded, elements)
    }

    /// Returns the im2col matrix through a specialized patch fill:
    /// contiguous runs of `kernel_width` copied per channel and kernel
    /// row, zero runs where the padding window leaves the input, and no
    /// per-element odometer arithmetic — the measured cost the fused
    /// window-GEMM pattern exists to remove. Non-contiguous inputs and
    /// non-dense storages take the composed reference path.
    ///
    /// # Panics
    /// Panics if `self` is not rank 4, `stride` is zero, or a kernel
    /// window does not fit the padded extents.
    fn windowed_patches(
        &self,
        kernel_height: usize,
        kernel_width: usize,
        stride: usize,
        padding: usize,
    ) -> Self {
        let shape = self.logical_shape();
        let axes = shape.axes();
        assert_eq!(
            axes.len(),
            4,
            "windowed_product input must be rank 4 [batch, channels, height, width], got {shape}"
        );
        assert!(stride > 0, "windowed_product stride must be positive");
        let (batch, channels, height, width) = (axes[0], axes[1], axes[2], axes[3]);
        assert!(
            kernel_height > 0
                && kernel_width > 0
                && kernel_height <= height + 2 * padding
                && kernel_width <= width + 2 * padding,
            "windowed_product kernel {kernel_height}x{kernel_width} does not fit {shape} \
             with padding {padding}"
        );
        let Some(elements) = self.as_slice() else {
            return composed_windowed_patches(self, kernel_height, kernel_width, stride, padding);
        };

        let out_height = (height + 2 * padding - kernel_height) / stride + 1;
        let out_width = (width + 2 * padding - kernel_width) / stride + 1;
        let columns = channels * kernel_height * kernel_width;
        let zero = elements[0].zero_like();
        let mut patches = vec![zero; batch * out_height * out_width * columns];
        for image in 0..batch {
            for out_y in 0..out_height {
                for out_x in 0..out_width {
                    let row = ((image * out_height + out_y) * out_width + out_x) * columns;
                    let source_x = (out_x * stride) as isize - padding as isize;
                    // The kernel columns that land inside the image; the
                    // rest of the patch row stays zero (the padding).
                    let clip_low = (-source_x).max(0) as usize;
                    let clip_high = kernel_width.min((width as isize - source_x).max(0) as usize);
                    if clip_low >= clip_high {
                        continue;
                    }
                    let run = clip_high - clip_low;
                    for channel in 0..channels {
                        for kernel_y in 0..kernel_height {
                            let source_y = (out_y * stride + kernel_y) as isize - padding as isize;
                            if source_y < 0 || source_y >= height as isize {
                                continue;
                            }
                            let source =
                                ((image * channels + channel) * height + source_y as usize) * width
                                    + (source_x + clip_low as isize) as usize;
                            let target = row
                                + (channel * kernel_height + kernel_y) * kernel_width
                                + clip_low;
                            patches[target..target + run]
                                .clone_from_slice(&elements[source..source + run]);
                        }
                    }
                }
            }
        }
        Self::dense(
            Shape::new([batch * out_height * out_width, columns]),
            patches,
        )
    }

    /// Returns the rows of `self` selected by `selection`, a one-hot
    /// `[count, vocab]` whose vocabulary must equal `self`'s first axis; the
    /// result is `[count, ...self.shape[1..]]` with row `i` equal to
    /// `self`'s row `selection_index(i)`.
    ///
    /// # Panics
    /// Panics if `selection` is not a `[count, vocab]` selection, `self` has
    /// no axes, or the vocabulary does not match `self`'s first axis.
    fn gather(&self, selection: &Self) -> Self {
        let table = self.logical_shape();
        let indices = selection.selection_indices();
        assert!(table.rank() >= 1, "gather table needs at least one axis");
        let vocabulary = selection.logical_shape().axes()[1];
        assert_eq!(
            vocabulary,
            table.axes()[0],
            "gather selection vocabulary {vocabulary} does not match table rows {}",
            table.axes()[0]
        );
        let row_size: usize = table.axes()[1..].iter().product();

        let mut elements = Vec::with_capacity(indices.len() * row_size);
        for &row in indices {
            for offset in 0..row_size {
                elements.push(self.get(row * row_size + offset));
            }
        }
        let result =
            Shape::new(std::iter::once(indices.len()).chain(table.axes()[1..].iter().copied()));
        Self::dense(result, elements)
    }

    /// Scatter-adds the rows of `self` (a `[count, ...]` gradient) into a
    /// zero `[rows, ...]` payload by `selection`'s indices: the adjoint of
    /// [`gather`](Tensorial::gather) and its gradient rule. Rows selected
    /// more than once accumulate.
    fn scatter(&self, selection: &Self, rows: usize) -> Self {
        let gradient = self.logical_shape();
        assert!(
            gradient.rank() >= 1,
            "scatter needs a gradient with a leading selection axis"
        );
        let indices = selection.selection_indices();
        assert_eq!(
            gradient.axes()[0],
            indices.len(),
            "scatter gradient rows disagree with the selection count"
        );
        assert_eq!(
            selection.logical_shape().axes()[1],
            rows,
            "scatter rows disagree with the selection vocabulary"
        );
        let row_size: usize = gradient.axes()[1..].iter().product();
        let volume = rows
            .checked_mul(row_size)
            .expect("shape volume overflows `usize`");
        let zero = self.get(0).zero_like();

        let mut elements = vec![zero; volume];
        for (source, &target) in indices.iter().enumerate() {
            for offset in 0..row_size {
                let position = target * row_size + offset;
                elements[position] =
                    elements[position].clone() + self.get(source * row_size + offset);
            }
        }
        let result = Shape::new(std::iter::once(rows).chain(gradient.axes()[1..].iter().copied()));
        Self::dense(result, elements)
    }
}

#[cfg(test)]
#[path = "tests/tensor_tests.rs"]
mod tests;
