use smallvec::smallvec;

use crate::{Elementary, Shape};

use super::{Cotangents, Operation, Retention, unary};

/// The rectified linear unit: the elementwise maximum of a value and
/// zero.
///
/// It is a dedicated unary variant rather than a `Maximum` against a
/// recorded zero because recording cannot construct a zero payload for a
/// generic `Data`, while the rule reaches one at run time through
/// `zero_like`. The gradient passes where the operand is non-negative
/// and stops elsewhere, routed by the 0/1 [`step`](Elementary::step)
/// indicator; the subgradient at zero is one, matching `Maximum`'s
/// left-biased tie rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Relu;

impl Relu {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the retention of the derivative rule below.
    /// It reads its operand to mask the active positions.
    pub(crate) fn retains(&self) -> Retention {
        Retention {
            operands: [true, false],
            output: false,
        }
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl<Data: Elementary> Operation<Data> for Relu {
    fn forward(&self, operands: &[&Data]) -> Data {
        let &operand = unary(operands);
        operand.maximum(&operand.zero_like())
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        smallvec![Some(gradient.clone() * operand.step(&operand.zero_like()))]
    }
}

#[cfg(test)]
#[path = "tests/relu_tests.rs"]
mod tests;
