use smallvec::smallvec;

use crate::{Elementary, Shape};

use super::{Cotangents, Operation, binary};

/// A value raised to the power of another, elementwise, with operands
/// `[base, exponent]`.
///
/// The base's gradient follows the power rule,
/// `exponent * base^(exponent - 1)`; the exponent's follows the
/// exponential rule, `output * ln(base)`, which is a number only on a
/// positive base — elsewhere the payload's `ln` semantics (`NaN` for
/// scalars) propagate, mirroring the mathematics: `x^y` has no exponent
/// derivative at a non-positive base.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Powf;

impl Powf {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (base, exponent) = binary(operands);
        assert_eq!(base, exponent, "powf requires operands of equal shapes");
        base.clone()
    }
}

impl<Data: Elementary> Operation<Data> for Powf {
    fn forward(&self, operands: &[&Data]) -> Data {
        let (&base, &exponent) = binary(operands);
        base.powf(exponent.clone())
    }

    fn backward(&self, operands: &[&Data], output: &Data, gradient: &Data) -> Cotangents<Data> {
        let (&base, &exponent) = binary(operands);
        let lowered = exponent.clone() - exponent.one_like();
        let base_cotangent = gradient.clone() * exponent.clone() * base.powf(lowered);
        let exponent_cotangent = gradient.clone() * output.clone() * base.ln();
        smallvec![Some(base_cotangent), Some(exponent_cotangent)]
    }
}

#[cfg(test)]
#[path = "tests/powf_tests.rs"]
mod tests;
