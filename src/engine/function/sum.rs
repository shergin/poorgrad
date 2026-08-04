use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Retention, unary};

/// The sum of every value in a payload, reduced to a single value.
///
/// Summation and broadcasting are adjoint: the gradient of the operand is
/// the incoming single-value gradient spread back across the operand's
/// shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sum;

impl Sum {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the retention of the derivative rule below.
    /// It reads its operand for shape only, which a placeholder answers.
    pub(crate) fn retains(&self) -> Retention {
        Retention::NOTHING
    }

    /// Infers the shape of the result: a rank-0 single value.
    pub(crate) fn infer_shape(&self, _operands: &[Shape]) -> Shape {
        Shape::scalar()
    }
}

impl<Data: Tensorial> Operation<Data> for Sum {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).sum()
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        smallvec![Some(gradient.broadcast_like(operand))]
    }
}
