use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::Shape;

/// The base payload contract for values in a computation graph.
///
/// Graph construction and gradient accumulation use the arithmetic operations
/// in this trait. The built-in implementations cover `f32`, `f64`, and
/// `Tensor<Element>` whenever its element type also implements
/// `Differentiable`.
///
/// Payloads must be `Send + Sync` because networks can be shared and evaluated
/// across threads. They need only be `Clone`, not `Copy`; cloning a tensor, for
/// example, shares its element buffer.
///
/// Implementations must keep shapes coherent. `zero_like`, `one_like`, and
/// negation preserve the operand's shape, and binary arithmetic on compatible
/// operands produces that same shape.
pub trait Differentiable:
    Clone
    + Debug
    + Send
    + Sync
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    /// Returns a zero shaped like `self`, used to seed gradient accumulators.
    ///
    /// It takes `&self` rather than being a nullary constructor so the identity
    /// can match the shape of the value it seeds. For a tensor payload the zero
    /// must have the same shape, which a shapeless `zero()` could not provide.
    fn zero_like(&self) -> Self;

    /// Returns a one shaped like `self`, used to seed the output gradient.
    ///
    /// The returned payload must have the same shape as `self`.
    fn one_like(&self) -> Self;

    /// Returns the shape of this payload: its extent along every axis.
    ///
    /// It is what record-time shape inference seeds leaves with. Scalars
    /// are rank 0.
    fn shape(&self) -> Shape;
}

impl Differentiable for f32 {
    fn zero_like(&self) -> Self {
        0.0
    }

    fn one_like(&self) -> Self {
        1.0
    }

    fn shape(&self) -> Shape {
        Shape::scalar()
    }
}

impl Differentiable for f64 {
    fn zero_like(&self) -> Self {
        0.0
    }

    fn one_like(&self) -> Self {
        1.0
    }

    fn shape(&self) -> Shape {
        Shape::scalar()
    }
}
