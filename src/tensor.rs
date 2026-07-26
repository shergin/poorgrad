use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use static_assertions::assert_impl_all;

use super::{Differentiable, Elementary, Shape, Tensorial, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Tensor<f64>: Send, Sync);

/// A dense, fixed-shape tensor payload with elementwise arithmetic.
///
/// It is the first non-scalar payload: a `Network<Tensor<f64>>` runs the
/// whole engine unchanged, every node carrying a tensor and every
/// operation applying elementwise. The shape and the elements live behind
/// `Arc`s, so cloning is O(1) — the engine clones payloads liberally
/// during gradient accumulation, and a payload must be cheap to copy by
/// design. Binary operations require identical shapes; implicit
/// broadcasting is deliberately absent. Tensor-native operations (matrix
/// multiplication, reductions) arrive with a later trait tier.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor<Element> {
    shape: Shape,
    elements: Arc<Vec<Element>>,
}

impl<Element: Differentiable> Tensor<Element> {
    /// Creates a tensor of `shape` from `elements` in row-major order.
    ///
    /// # Panics
    /// Panics if the number of elements differs from the shape's volume.
    pub fn new(shape: impl IntoIterator<Item = usize>, elements: impl Into<Vec<Element>>) -> Self {
        let shape = Shape::new(shape);
        let elements = elements.into();
        assert_eq!(
            shape.volume(),
            elements.len(),
            "tensor shape does not match its number of elements"
        );
        Self {
            shape,
            elements: Arc::new(elements),
        }
    }

    /// Creates a tensor of `shape` with every element set to `element`.
    pub fn filled(shape: impl IntoIterator<Item = usize>, element: Element) -> Self {
        let shape = Shape::new(shape);
        let volume = shape.volume();
        Self {
            shape,
            elements: Arc::new(vec![element; volume]),
        }
    }

    /// Returns the elements in row-major order.
    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    /// Returns a tensor with every element passed through `transform`,
    /// sharing this tensor's shape.
    fn map(&self, transform: impl Fn(&Element) -> Element) -> Self {
        Self {
            shape: self.shape.clone(),
            elements: Arc::new(self.elements.iter().map(transform).collect()),
        }
    }

    /// Combines two tensors element by element with `combine`.
    ///
    /// # Panics
    /// Panics if the tensors have different shapes.
    fn zip(&self, other: &Self, combine: impl Fn(&Element, &Element) -> Element) -> Self {
        assert_eq!(self.shape, other.shape, "tensors have different shapes");
        Self {
            shape: self.shape.clone(),
            elements: Arc::new(
                self.elements
                    .iter()
                    .zip(other.elements.iter())
                    .map(|(left, right)| combine(left, right))
                    .collect(),
            ),
        }
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
    fn zero_like(&self) -> Self {
        self.map(|element| element.zero_like())
    }

    fn one_like(&self) -> Self {
        self.map(|element| element.one_like())
    }

    fn shape(&self) -> Shape {
        self.shape.clone()
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
    /// # Panics
    /// Panics if either operand is not rank 2, the inner dimensions do not
    /// agree, or any dimension is empty.
    fn matmul(&self, rhs: &Self) -> Self {
        assert_eq!(self.shape.rank(), 2, "matmul requires rank-2 tensors");
        assert_eq!(rhs.shape.rank(), 2, "matmul requires rank-2 tensors");
        let (rows, inner) = (self.shape.axes()[0], self.shape.axes()[1]);
        let (rhs_inner, columns) = (rhs.shape.axes()[0], rhs.shape.axes()[1]);
        assert_eq!(inner, rhs_inner, "matmul inner dimensions do not agree");
        assert!(
            rows > 0 && inner > 0 && columns > 0,
            "matmul requires non-empty dimensions"
        );

        let mut elements = Vec::with_capacity(rows * columns);
        for row in 0..rows {
            for column in 0..columns {
                let mut total = self.elements[row * inner].clone() * rhs.elements[column].clone();
                for step in 1..inner {
                    total = total
                        + self.elements[row * inner + step].clone()
                            * rhs.elements[step * columns + column].clone();
                }
                elements.push(total);
            }
        }
        Self {
            shape: Shape::new([rows, columns]),
            elements: Arc::new(elements),
        }
    }

    /// Returns the tensor with its two axes swapped.
    ///
    /// Rank-0 and rank-1 tensors are returned unchanged.
    ///
    /// # Panics
    /// Panics if the tensor's rank exceeds 2.
    fn transposed(&self) -> Self {
        if self.shape.rank() < 2 {
            return self.clone();
        }
        assert_eq!(self.shape.rank(), 2, "transpose supports rank 2 at most");
        let (rows, columns) = (self.shape.axes()[0], self.shape.axes()[1]);
        let mut elements = Vec::with_capacity(rows * columns);
        for column in 0..columns {
            for row in 0..rows {
                elements.push(self.elements[row * columns + column].clone());
            }
        }
        Self {
            shape: Shape::new([columns, rows]),
            elements: Arc::new(elements),
        }
    }

    /// Returns the sum of every element as a rank-0 tensor.
    ///
    /// # Panics
    /// Panics if the tensor holds no elements.
    fn sum(&self) -> Self {
        let mut elements = self.elements.iter();
        let first = elements
            .next()
            .expect("sum requires a non-empty tensor")
            .clone();
        let total = elements.fold(first, |total, element| total + element.clone());
        Self {
            shape: Shape::scalar(),
            elements: Arc::new(vec![total]),
        }
    }

    /// Returns this tensor's single element spread across `reference`'s
    /// shape: the one explicit form of broadcasting.
    ///
    /// # Panics
    /// Panics if `self` holds more than one element.
    fn broadcast_like(&self, reference: &Self) -> Self {
        assert_eq!(
            self.elements.len(),
            1,
            "broadcast requires a single-element tensor"
        );
        Self {
            shape: reference.shape.clone(),
            elements: Arc::new(vec![self.elements[0].clone(); reference.elements.len()]),
        }
    }
}

// `Tensor` is a local type, so unlike the foreign scalars in `value.rs`
// it gets the reversed literal operators generically.
impl<'network, Element: Differentiable> Add<Value<'network, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'network, Tensor<Element>>;

    fn add(self, rhs: Value<'network, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) + rhs
    }
}

impl<'network, Element: Differentiable> Sub<Value<'network, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'network, Tensor<Element>>;

    fn sub(self, rhs: Value<'network, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) - rhs
    }
}

impl<'network, Element: Differentiable> Mul<Value<'network, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'network, Tensor<Element>>;

    fn mul(self, rhs: Value<'network, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) * rhs
    }
}

impl<'network, Element: Differentiable> Div<Value<'network, Tensor<Element>>> for Tensor<Element> {
    type Output = Value<'network, Tensor<Element>>;

    fn div(self, rhs: Value<'network, Tensor<Element>>) -> Self::Output {
        rhs.literal(self) / rhs
    }
}

#[cfg(test)]
#[path = "tests/tensor_tests.rs"]
mod tests;
