use smallvec::smallvec;

use crate::{Differentiable, Shape};

use super::{Cotangents, Operation, Retention, unary};

/// The negation of a value.
///
/// The derivative with respect to the operand is minus one, so `backward`
/// hands the negated incoming gradient to the operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Neg;

impl Neg {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the retention of the derivative rule below.
    /// It reads no payloads.
    pub(crate) fn retains(&self) -> Retention {
        Retention::NOTHING
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Neg {
    fn forward(&self, operands: &[&Data]) -> Data {
        let &operand = unary(operands);
        -operand.clone()
    }

    fn backward(&self, _operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        smallvec![Some(-gradient.clone())]
    }
}
