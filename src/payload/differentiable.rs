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
    /// The type accumulating operations compute in before rounding
    /// back to `Self` once: matmul inner products, the sum
    /// reductions, `fold`, and the scatter adjoint promote every
    /// term, accumulate here, and demote the final total.
    ///
    /// `Self` for payloads that accumulate in their own precision;
    /// `f32` for `Bf16`, whose eight significand bits swamp once a
    /// total reaches 256 times a term. The choice is semantics, not
    /// an optimization: every representation and every path honors
    /// it — a constant operand accumulates exactly like a dense one —
    /// and StableHLO emission states it through
    /// `Emittable::ACCUMULATION`.
    type Accumulator: Clone
        + Debug
        + Send
        + Sync
        + Add<Output = Self::Accumulator>
        + Mul<Output = Self::Accumulator>;

    /// Returns this value in the accumulator type, exactly.
    fn promote(&self) -> Self::Accumulator;

    /// Returns an accumulated total rounded back into `Self`.
    fn demote(accumulated: Self::Accumulator) -> Self;

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

    /// Returns a payload of `shape` with every element equal to `count`.
    ///
    /// It is the constructor behind size-derived constants: a composed
    /// formula that divides by an axis extent (a mean, a normalization)
    /// must mint that extent as a payload. Unlike `zero_like` and
    /// `one_like` it cannot borrow a payload to copy the shape from,
    /// because a composite over computed values has no payload at hand.
    /// Counts convert exactly as long as the payload's numeric type can
    /// represent them.
    fn counted(shape: Shape, count: usize) -> Self;

    /// Returns the shape of this payload: its extent along every axis.
    ///
    /// It is what record-time shape inference seeds leaves with. Scalars
    /// are rank 0.
    fn shape(&self) -> Shape;

    /// Returns whether this payload is exactly what
    /// [`counted`](Differentiable::counted) mints for `shape` and
    /// `count`.
    ///
    /// It is the recognizer half of `counted`: pattern matchers use it
    /// to certify a recorded size-derived constant (the divisor of a
    /// composed mean) before raising the surrounding formula to a
    /// named target operation, where an unverified divisor would
    /// silently change semantics abroad. The conservative default
    /// answers `false`, which only forgoes recognitions: a payload
    /// that cannot certify stays on the primitive path.
    fn is_counted(&self, shape: &Shape, count: usize) -> bool {
        let _ = (shape, count);
        false
    }
}

impl Differentiable for f32 {
    type Accumulator = Self;

    fn promote(&self) -> Self {
        *self
    }

    fn demote(accumulated: Self) -> Self {
        accumulated
    }

    fn zero_like(&self) -> Self {
        0.0
    }

    fn one_like(&self) -> Self {
        1.0
    }

    /// Scalar payloads ignore the requested shape, mirroring the
    /// identity semantics of their `Tensorial` operations.
    fn counted(_shape: Shape, count: usize) -> Self {
        count as f32
    }

    fn shape(&self) -> Shape {
        Shape::scalar()
    }

    /// Scalar payloads ignore the shape, mirroring `counted`.
    fn is_counted(&self, _shape: &Shape, count: usize) -> bool {
        *self == count as f32
    }
}

impl Differentiable for f64 {
    type Accumulator = Self;

    fn promote(&self) -> Self {
        *self
    }

    fn demote(accumulated: Self) -> Self {
        accumulated
    }

    fn zero_like(&self) -> Self {
        0.0
    }

    fn one_like(&self) -> Self {
        1.0
    }

    /// Scalar payloads ignore the requested shape, mirroring the
    /// identity semantics of their `Tensorial` operations.
    fn counted(_shape: Shape, count: usize) -> Self {
        count as f64
    }

    fn shape(&self) -> Shape {
        Shape::scalar()
    }

    /// Scalar payloads ignore the shape, mirroring `counted`.
    fn is_counted(&self, _shape: &Shape, count: usize) -> bool {
        *self == count as f64
    }
}
