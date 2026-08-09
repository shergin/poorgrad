use std::fmt::{self, Debug};
use std::ops::{Add, Div, Mul, Neg, Sub};

use super::{Differentiable, Elementary, Shape, Tensorial};

/// A brain-float 16 payload: the top half of an `f32`, one sign bit,
/// eight exponent bits, and seven stored mantissa bits.
///
/// Every operation converts to `f32`, computes there, and rounds the
/// result back to the nearest bf16 (ties to even) — the standard bf16
/// semantic, deterministic on every platform. Integers are exact up
/// to 256; above that the significand steps by two, then four, and so
/// on, which bounds the [`Differentiable::counted`] contract.
///
/// Matrix multiplication follows this per-op semantic on the composed
/// path; the accumulation contract is pinned separately at the
/// [`Elementary::gemm`] seam.
#[derive(Clone, Copy)]
pub struct Bf16(u16);

impl Bf16 {
    /// The additive identity.
    pub const ZERO: Self = Self(0x0000);

    /// The multiplicative identity.
    pub const ONE: Self = Self(0x3F80);

    /// Returns the nearest bf16 to `value`, rounding ties to even.
    ///
    /// The carry from the rounding bias propagates into the exponent,
    /// so values beyond the finite range round to infinity, exactly
    /// as IEEE round-to-nearest requires. A NaN stays a NaN: the top
    /// half is kept and a mantissa bit is forced on so the payload
    /// cannot truncate to an infinity.
    pub fn from_f32(value: f32) -> Self {
        let bits = value.to_bits();
        if value.is_nan() {
            return Self(((bits >> 16) | 0x0040) as u16);
        }
        let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
        Self(((bits + rounding_bias) >> 16) as u16)
    }

    /// Returns the exact `f32` this bf16 denotes: the bits shifted
    /// into the top half of a single. Every bf16 value is exactly
    /// representable, so this conversion never rounds.
    pub fn to_f32(self) -> f32 {
        f32::from_bits((self.0 as u32) << 16)
    }

    /// Returns the raw bit pattern.
    pub fn to_bits(self) -> u16 {
        self.0
    }

    /// Returns the bf16 with the raw bit pattern `bits`.
    pub fn from_bits(bits: u16) -> Self {
        Self(bits)
    }
}

impl From<f32> for Bf16 {
    fn from(value: f32) -> Self {
        Self::from_f32(value)
    }
}

impl From<Bf16> for f32 {
    fn from(value: Bf16) -> f32 {
        value.to_f32()
    }
}

/// Value equality with float semantics: positive and negative zero
/// are equal and a NaN equals nothing, matching `f32` rather than
/// the bit pattern.
impl PartialEq for Bf16 {
    fn eq(&self, other: &Self) -> bool {
        self.to_f32() == other.to_f32()
    }
}

impl Debug for Bf16 {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(&self.to_f32(), formatter)
    }
}

impl Add for Bf16 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() + rhs.to_f32())
    }
}

impl Sub for Bf16 {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() - rhs.to_f32())
    }
}

impl Mul for Bf16 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() * rhs.to_f32())
    }
}

impl Div for Bf16 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        Self::from_f32(self.to_f32() / rhs.to_f32())
    }
}

impl Neg for Bf16 {
    type Output = Self;

    /// Negation flips the sign bit directly: exact for every value,
    /// including zeros, infinities, and NaN, with no round trip.
    fn neg(self) -> Self {
        Self(self.0 ^ 0x8000)
    }
}

impl Differentiable for Bf16 {
    fn zero_like(&self) -> Self {
        Self::ZERO
    }

    fn one_like(&self) -> Self {
        Self::ONE
    }

    /// Scalar payloads ignore the requested shape, mirroring the
    /// identity semantics of their `Tensorial` operations. Counts
    /// are exact up to 256, the last integer bf16 represents
    /// exactly; larger counts round to nearest even.
    fn counted(_shape: Shape, count: usize) -> Self {
        Self::from_f32(count as f32)
    }

    fn shape(&self) -> Shape {
        Shape::scalar()
    }
}

impl Elementary for Bf16 {
    fn exp(&self) -> Self {
        Self::from_f32(self.to_f32().exp())
    }

    fn ln(&self) -> Self {
        Self::from_f32(self.to_f32().ln())
    }

    fn sqrt(&self) -> Self {
        Self::from_f32(self.to_f32().sqrt())
    }

    fn tanh(&self) -> Self {
        Self::from_f32(self.to_f32().tanh())
    }

    fn powf(&self, exponent: Self) -> Self {
        Self::from_f32(self.to_f32().powf(exponent.to_f32()))
    }

    /// The result is one of the operands, both exactly representable,
    /// so the round trip through `f32` never rounds.
    fn maximum(&self, other: &Self) -> Self {
        Self::from_f32(self.to_f32().max(other.to_f32()))
    }

    fn step(&self, threshold: &Self) -> Self {
        if self.to_f32() >= threshold.to_f32() {
            Self::ONE
        } else {
            Self::ZERO
        }
    }
}

impl Tensorial for Bf16 {
    /// Scalar payloads use identity semantics: the patches are the
    /// value itself, so the product degenerates to the scalar matmul.
    fn windowed_patches(
        &self,
        _kernel_height: usize,
        _kernel_width: usize,
        _stride: usize,
        _padding: usize,
    ) -> Self {
        *self
    }

    fn matmul(&self, rhs: &Self) -> Self {
        *self * *rhs
    }

    fn transpose(&self) -> Self {
        *self
    }

    fn sum(&self) -> Self {
        *self
    }

    fn sum_along(&self, _axis: usize) -> Self {
        *self
    }

    fn max_along(&self, _axis: usize) -> Self {
        *self
    }

    fn broadcast_like(&self, _reference: &Self) -> Self {
        *self
    }

    fn broadcast_along(&self, _axis: usize, _reference: &Self) -> Self {
        *self
    }

    fn reshape(&self, shape: Shape) -> Self {
        // The one movement request a scalar graph can record (volumes
        // match), so the capability mismatch is rejected here rather
        // than silently breaking recorded/payload shape coherence.
        assert_eq!(
            shape.rank(),
            0,
            "a scalar payload cannot take shape {shape}"
        );
        *self
    }

    fn permute(&self, _order: &[usize]) -> Self {
        *self
    }

    fn narrow(&self, _axis: usize, _start: usize, _len: usize) -> Self {
        *self
    }

    fn pad(&self, _axis: usize, _start: usize, _full_extent: usize) -> Self {
        *self
    }

    fn unfold(&self, _axis: usize, _size: usize, _step: usize, _dilation: usize) -> Self {
        *self
    }

    fn fold(
        &self,
        _axis: usize,
        _size: usize,
        _step: usize,
        _dilation: usize,
        _extent: usize,
    ) -> Self {
        *self
    }

    fn gather(&self, _selection: &Self) -> Self {
        *self
    }

    fn scatter(&self, _selection: &Self, _rows: usize) -> Self {
        *self
    }
}

#[cfg(test)]
#[path = "tests/bf16_tests.rs"]
mod tests;
