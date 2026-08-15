use smallvec::smallvec;

use crate::{Elementary, Shape};

use super::{Cotangents, Operation, Reads, unary};

/// The natural logarithm of a value.
///
/// The derivative of `ln(x)` is `1 / x`, so `backward` divides the
/// incoming gradient by the operand's value. Gradients inherit the
/// payload's logarithm and division semantics outside the positive
/// domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ln;

impl Ln {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its operand: the derivative divides by it.
    pub(crate) fn reads(&self) -> Reads {
        Reads {
            operands: [true, false],
            output: false,
        }
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl<Data: Elementary> Operation<Data> for Ln {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).ln()
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        smallvec![Some(gradient.clone() / operand.clone())]
    }
}
