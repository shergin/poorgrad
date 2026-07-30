use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use static_assertions::assert_impl_all;

use super::layout::{Layout, Strides};
use super::storage::Storage;
use super::{Differentiable, Elementary, Shape, Tensorial};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Tensor<f64>: Send, Sync);

/// A dense tensor with an immutable, runtime-defined [`Shape`] and a shared
/// element buffer read through a strided [`layout`](super::layout::Layout).
///
/// The elements are held behind a [`Storage`] representation: an
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
/// [`Tensorial::transposed`] accepts ranks 0 through 2, returning a view.
/// Reductions and explicit broadcasts are rank-general. Reshaping and
/// batched matrix multiplication are not supported.
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
        }
    }

    /// Returns the element at logical row-major `position`.
    ///
    /// It is the general per-element read shared by every operation; a
    /// dense layout resolves it through the stride and offset arithmetic,
    /// and a constant answers with its single value.
    fn get(&self, position: usize) -> &Element {
        match &self.storage {
            Storage::Dense { data, layout } => &data[layout.storage_index(position)],
            Storage::Constant { value, .. } => value,
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

    /// Returns the elements in logical row-major order.
    ///
    /// A contiguous dense buffer iterates its slice directly; a strided
    /// view walks its layout with an odometer; a constant repeats its
    /// single value across the shape's volume.
    pub fn iter(&self) -> impl Iterator<Item = &Element> + '_ {
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
        }
    }
}

impl<Element: Clone> Tensor<Element> {
    /// Returns the elements in logical row-major order as an owned vector.
    pub fn to_vec(&self) -> Vec<Element> {
        self.iter().cloned().collect()
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
    pub fn new(shape: impl IntoIterator<Item = usize>, elements: impl Into<Vec<Element>>) -> Self {
        let shape = Shape::new(shape);
        let elements = elements.into();
        assert_eq!(
            shape.volume(),
            elements.len(),
            "tensor shape does not match its number of elements"
        );
        assert!(
            !elements.is_empty(),
            "tensors must hold at least one element"
        );
        Self::dense(shape, elements)
    }

    /// Creates a tensor of `shape` with every element set to `element`,
    /// stored as a non-allocating constant.
    ///
    /// # Panics
    /// Panics if the shape's volume overflows `usize` or the shape holds no
    /// elements, as documented on [`Tensor::new`].
    pub fn filled(shape: impl IntoIterator<Item = usize>, element: Element) -> Self {
        let shape = Shape::new(shape);
        assert!(shape.volume() > 0, "tensors must hold at least one element");
        Self::constant(shape, element)
    }

    /// Builds a contiguous dense tensor of `shape` from `elements`, without
    /// the public constructor's validation.
    fn dense(shape: Shape, elements: Vec<Element>) -> Self {
        Self {
            storage: Storage::Dense {
                layout: Layout::contiguous(shape),
                data: Arc::new(elements),
            },
        }
    }

    /// Builds a constant tensor of `shape` filled with `value`.
    fn constant(shape: Shape, value: Element) -> Self {
        Self {
            storage: Storage::Constant { shape, value },
        }
    }

    /// Returns a tensor with every element passed through `transform`.
    ///
    /// A constant maps in place to another constant; a dense tensor
    /// materializes the transformed elements in logical order.
    fn map(&self, transform: impl Fn(&Element) -> Element) -> Self {
        match &self.storage {
            Storage::Constant { shape, value } => Self::constant(shape.clone(), transform(value)),
            Storage::Dense { .. } => Self::dense(
                self.logical_shape().clone(),
                self.iter().map(transform).collect(),
            ),
        }
    }

    /// Combines two tensors element by element with `combine`.
    ///
    /// Two constants combine into a constant; otherwise the result is a
    /// dense tensor built in logical order.
    ///
    /// # Panics
    /// Panics if the tensors have different shapes.
    fn zip(&self, other: &Self, combine: impl Fn(&Element, &Element) -> Element) -> Self {
        assert_eq!(
            self.logical_shape(),
            other.logical_shape(),
            "tensors have different shapes"
        );
        match (&self.storage, &other.storage) {
            (Storage::Constant { value: left, .. }, Storage::Constant { value: right, .. }) => {
                Self::constant(self.logical_shape().clone(), combine(left, right))
            }
            _ => Self::dense(
                self.logical_shape().clone(),
                self.iter()
                    .zip(other.iter())
                    .map(|(left, right)| combine(left, right))
                    .collect(),
            ),
        }
    }
}

/// Returns the shape with its two axes swapped, matching the payload
/// transpose that a constant undergoes without touching a buffer.
///
/// # Panics
/// Panics if the rank exceeds 2.
fn transposed_shape(shape: &Shape) -> Shape {
    if shape.rank() < 2 {
        return shape.clone();
    }
    assert_eq!(shape.rank(), 2, "transpose supports rank 2 at most");
    let axes = shape.axes();
    Shape::new([axes[1], axes[0]])
}

/// Iterator over a tensor's elements in logical row-major order.
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
}

impl<'tensor, Element> Iterator for ElementIter<'tensor, Element> {
    type Item = &'tensor Element;

    fn next(&mut self) -> Option<&'tensor Element> {
        match self {
            ElementIter::Constant { value, remaining } => {
                if *remaining == 0 {
                    return None;
                }
                *remaining -= 1;
                Some(*value)
            }
            ElementIter::Contiguous(iterator) => iterator.next(),
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
                let slice: &'tensor [Element] = data;
                let element = &slice[*index];
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
        }
    }
}

impl<Element: PartialEq> PartialEq for Tensor<Element> {
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

    fn shape(&self) -> Shape {
        self.logical_shape().clone()
    }
}

impl<Element: Elementary> Elementary for Tensor<Element> {
    fn exp(&self) -> Self {
        self.map(|element| element.exp())
    }

    fn ln(&self) -> Self {
        self.map(|element| element.ln())
    }

    fn tanh(&self) -> Self {
        self.map(|element| element.tanh())
    }

    fn powf(&self, exponent: Self) -> Self {
        self.zip(&exponent, |element, exponent| {
            element.powf(exponent.clone())
        })
    }
}

impl<Element: Elementary> Tensorial for Tensor<Element> {
    /// Returns the matrix product of two rank-2 tensors.
    ///
    /// Operands are read through logical access, so strided views (a
    /// transposed operand, most often) multiply without first being
    /// materialized.
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

        let mut elements = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                let mut total = self.get(row * inner).clone() * rhs.get(column).clone();
                for step in 1..inner {
                    total = total
                        + self.get(row * inner + step).clone()
                            * rhs.get(step * columns + column).clone();
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
    fn transposed(&self) -> Self {
        match &self.storage {
            Storage::Dense { data, layout } => Self {
                storage: Storage::Dense {
                    data: Arc::clone(data),
                    layout: layout.transposed(),
                },
            },
            Storage::Constant { shape, value } => {
                Self::constant(transposed_shape(shape), value.clone())
            }
        }
    }

    /// Returns the sum of every element as a rank-0 constant.
    ///
    /// Elements are accumulated in logical order from left to right without
    /// pairwise or compensated summation.
    fn sum(&self) -> Self {
        let mut elements = self.iter();
        let first = elements
            .next()
            .expect("sum requires a non-empty tensor")
            .clone();
        let total = elements.fold(first, |total, element| total + element.clone());
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
                let mut total = self.get(position(0)).clone();
                for step in 1..extent {
                    total = total + self.get(position(step)).clone();
                }
                elements.push(total);
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
        Self::constant(reference.logical_shape().clone(), self.get(0).clone())
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
        }
    }

    /// Returns `self` reinterpreted with `shape` in logical row-major
    /// order.
    ///
    /// A contiguous dense tensor and a constant reshape into an O(1) view
    /// over the same buffer; a strided view is first materialized.
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
            Storage::Dense { data, layout } => match layout.reshaped(shape.clone()) {
                Some(reshaped) => Self {
                    storage: Storage::Dense {
                        data: Arc::clone(data),
                        layout: reshaped,
                    },
                },
                None => Self::dense(shape, self.to_vec()),
            },
        }
    }

    /// Returns `self` with its axes reordered by `order` as a view over the
    /// same buffer.
    ///
    /// # Panics
    /// Panics if `order` is not a permutation of `0..rank`.
    fn permuted(&self, order: &[usize]) -> Self {
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
                    layout: layout.permuted(order),
                },
            },
        }
    }

    /// Returns the window of `len` elements from `start` along `axis` as a
    /// view over the same buffer.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank or `start + len` exceeds its extent.
    fn narrowed(&self, axis: usize, start: usize, len: usize) -> Self {
        let shape = self.logical_shape();
        assert!(
            axis < shape.rank(),
            "narrow axis {axis} is out of rank for {shape}"
        );
        let extent = shape.axes()[axis];
        assert!(
            start + len <= extent,
            "narrow window {start}..{} exceeds axis {axis} extent {extent}",
            start + len
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
                    layout: layout.narrowed(axis, start, len),
                },
            },
        }
    }

    /// Returns `self` placed at `start ..` along `axis` inside a tensor
    /// whose `axis` has extent `full_extent`, with zeros elsewhere.
    ///
    /// # Panics
    /// Panics if `axis` is out of rank or the window exceeds `full_extent`.
    fn padded(&self, axis: usize, start: usize, full_extent: usize) -> Self {
        let shape = self.logical_shape();
        assert!(
            axis < shape.rank(),
            "pad axis {axis} is out of rank for {shape}"
        );
        let axes = shape.axes();
        let len = axes[axis];
        assert!(
            start + len <= full_extent,
            "pad window {start}..{} exceeds the full extent {full_extent}",
            start + len
        );
        let outer: usize = axes[..axis].iter().product();
        let inner: usize = axes[axis + 1..].iter().product();
        let zero = self.get(0).zero_like();

        let mut elements = Vec::with_capacity(outer * full_extent * inner);
        for outer_index in 0..outer {
            for position in 0..full_extent {
                for inner_index in 0..inner {
                    if position >= start && position < start + len {
                        let source = (outer_index * len + (position - start)) * inner + inner_index;
                        elements.push(self.get(source).clone());
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
}

#[cfg(test)]
#[path = "tests/tensor_tests.rs"]
mod tests;
