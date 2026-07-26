use std::fmt::Debug;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// The numeric payload a `Value` node carries through the computation graph.
///
/// It captures exactly the operations the autograd engine requires of an
/// underlying value, so `Value` can be generic over `f32`, `f64`, and later
/// tensor types without the engine depending on any of them directly.
///
/// It requires `Send + Sync` on purpose: the premise of the engine is a graph
/// that can be shared and evaluated across threads, so every payload must be
/// shareable too. It requires only `Clone`, never `Copy`, so that non-`Copy`
/// payloads such as tensors can implement it later without a breaking change.
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
    fn one_like(&self) -> Self;
}

impl Differentiable for f32 {
    fn zero_like(&self) -> Self {
        0.0
    }

    fn one_like(&self) -> Self {
        1.0
    }
}

impl Differentiable for f64 {
    fn zero_like(&self) -> Self {
        0.0
    }

    fn one_like(&self) -> Self {
        1.0
    }
}
