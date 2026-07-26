use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::Arc;

use static_assertions::assert_impl_all;

use super::{Differentiable, Elementary};

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
    shape: Arc<Vec<usize>>,
    elements: Arc<Vec<Element>>,
}

impl<Element: Differentiable> Tensor<Element> {
    /// Creates a tensor of `shape` from `elements` in row-major order.
    ///
    /// # Panics
    /// Panics if the number of elements differs from the shape's volume.
    pub fn new(shape: impl Into<Vec<usize>>, elements: impl Into<Vec<Element>>) -> Self {
        let shape = shape.into();
        let elements = elements.into();
        let volume: usize = shape.iter().product();
        assert_eq!(
            volume,
            elements.len(),
            "tensor shape does not match its number of elements"
        );
        Self {
            shape: Arc::new(shape),
            elements: Arc::new(elements),
        }
    }

    /// Creates a tensor of `shape` with every element set to `element`.
    pub fn filled(shape: impl Into<Vec<usize>>, element: Element) -> Self {
        let shape = shape.into();
        let volume = shape.iter().product();
        Self {
            shape: Arc::new(shape),
            elements: Arc::new(vec![element; volume]),
        }
    }

    /// Returns the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the elements in row-major order.
    pub fn elements(&self) -> &[Element] {
        &self.elements
    }

    /// Returns a tensor with every element passed through `transform`,
    /// sharing this tensor's shape.
    fn map(&self, transform: impl Fn(&Element) -> Element) -> Self {
        Self {
            shape: Arc::clone(&self.shape),
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
            shape: Arc::clone(&self.shape),
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

#[cfg(test)]
#[path = "tests/tensor_tests.rs"]
mod tests;
