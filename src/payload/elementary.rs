use super::Differentiable;

/// Elementary numeric functions supported by graph payloads.
///
/// This trait extends [`Differentiable`] without making transcendental
/// functions or order comparisons part of the base arithmetic contract. The
/// scalar implementations use the corresponding `f32` and `f64` operations;
/// `Tensor<Element>` applies them elementwise when `Element` also implements
/// `Elementary`.
pub trait Elementary: Differentiable {
    /// Returns `e` raised to the power of `self`.
    fn exp(&self) -> Self;

    /// Returns the natural logarithm of `self`.
    fn ln(&self) -> Self;

    /// Returns the hyperbolic tangent of `self`.
    fn tanh(&self) -> Self;

    /// Returns `self` raised to the power of `exponent`.
    fn powf(&self, exponent: Self) -> Self;

    /// Returns the elementwise maximum of `self` and `other`.
    ///
    /// It is the payload-returning form of comparison: a `bool` answer
    /// could not express an elementwise result, so order enters the
    /// contract as an operation rather than as `PartialOrd`.
    fn maximum(&self, other: &Self) -> Self;
}

impl Elementary for f32 {
    fn exp(&self) -> Self {
        f32::exp(*self)
    }

    fn ln(&self) -> Self {
        f32::ln(*self)
    }

    fn tanh(&self) -> Self {
        f32::tanh(*self)
    }

    fn powf(&self, exponent: Self) -> Self {
        f32::powf(*self, exponent)
    }

    fn maximum(&self, other: &Self) -> Self {
        f32::max(*self, *other)
    }
}

impl Elementary for f64 {
    fn exp(&self) -> Self {
        f64::exp(*self)
    }

    fn ln(&self) -> Self {
        f64::ln(*self)
    }

    fn tanh(&self) -> Self {
        f64::tanh(*self)
    }

    fn powf(&self, exponent: Self) -> Self {
        f64::powf(*self, exponent)
    }

    fn maximum(&self, other: &Self) -> Self {
        f64::max(*self, *other)
    }
}
