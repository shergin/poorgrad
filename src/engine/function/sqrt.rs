use smallvec::smallvec;

use crate::{Elementary, Shape};

use super::{Cotangents, Operation, Retention, unary};

/// The square root of a value.
///
/// The derivative of `sqrt(x)` is `1 / (2 * sqrt(x))`, so `backward`
/// divides the incoming gradient by twice the node's own output — no
/// generic literal `2` exists, so the doubling is `output + output`.
/// Gradients inherit the payload's root and division semantics outside
/// the positive domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sqrt;

impl Sqrt {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the retention of the derivative rule below.
    /// It reads its own output: the derivative divides by twice it.
    pub(crate) fn retains(&self) -> Retention {
        Retention {
            operands: [false, false],
            output: true,
        }
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl<Data: Elementary> Operation<Data> for Sqrt {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).sqrt()
    }

    fn backward(&self, _operands: &[&Data], output: &Data, gradient: &Data) -> Cotangents<Data> {
        smallvec![Some(gradient.clone() / (output.clone() + output.clone()))]
    }
}

#[cfg(test)]
#[path = "tests/sqrt_tests.rs"]
mod tests;
