use crate::backend;

use super::Differentiable;
use super::gemm::GemmTask;

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

    /// Returns the square root of `self`.
    ///
    /// It is a distinct operation rather than `powf(0.5)` because IEEE 754
    /// requires `sqrt` to be correctly rounded and makes no such promise
    /// for `pow`.
    fn sqrt(&self) -> Self;

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

    /// Returns the elementwise 0/1 indicator of `self >= threshold`: the
    /// Heaviside step, one where `self` reaches the threshold and zero
    /// elsewhere.
    ///
    /// It carries the derivative of the `maximum` family, marking the
    /// positions where the left side won; ties answer one.
    fn step(&self, threshold: &Self) -> Self;

    /// Offers a matrix-multiplication task to the compiled backend
    /// chain: the acceleration seam.
    ///
    /// It answers `None` — compute on the built-in paths — unless the
    /// element type has a backend entry point; `f32` and `f64`
    /// forward to the chain in `backend`. Leave the default unless
    /// you are routing to a kernel; answering `Some` asserts the
    /// row-major product of exactly the described task.
    fn gemm(task: &GemmTask<'_, Self>) -> Option<Vec<Self>>
    where
        Self: Sized,
    {
        let _ = task;
        None
    }
}

impl Elementary for f32 {
    fn exp(&self) -> Self {
        f32::exp(*self)
    }

    fn ln(&self) -> Self {
        f32::ln(*self)
    }

    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
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

    fn step(&self, threshold: &Self) -> Self {
        if *self >= *threshold { 1.0 } else { 0.0 }
    }

    fn gemm(task: &GemmTask<'_, Self>) -> Option<Vec<Self>> {
        backend::gemm_f32(task)
    }
}

impl Elementary for f64 {
    fn exp(&self) -> Self {
        f64::exp(*self)
    }

    fn ln(&self) -> Self {
        f64::ln(*self)
    }

    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
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

    fn step(&self, threshold: &Self) -> Self {
        if *self >= *threshold { 1.0 } else { 0.0 }
    }

    fn gemm(task: &GemmTask<'_, Self>) -> Option<Vec<Self>> {
        backend::gemm_f64(task)
    }
}
