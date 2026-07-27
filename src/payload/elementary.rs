use super::Differentiable;

/// The elementary transcendental functions used by activation `Function`s.
///
/// It extends `Differentiable` with the nonlinear operations activations need,
/// kept separate from the core trait so the engine's arithmetic does not force
/// every payload to implement functions it may not support.
pub trait Elementary: Differentiable {
    /// Returns `e` raised to the power of `self`.
    fn exp(&self) -> Self;

    /// Returns the natural logarithm of `self`.
    fn ln(&self) -> Self;

    /// Returns the hyperbolic tangent of `self`.
    fn tanh(&self) -> Self;

    /// Returns `self` raised to the power of `exponent`.
    fn powf(&self, exponent: Self) -> Self;
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
}
