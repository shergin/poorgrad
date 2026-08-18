use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// The sum of a payload along one named axis.
///
/// It is the axis-wise form of `Sum`, and `BroadcastAlong` is its
/// adjoint: the operand's gradient is the incoming gradient repeated
/// back along the reduced axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SumAlong {
    pub(crate) axis: usize,
}

impl SumAlong {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its operand for shape only, which a placeholder answers.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the shape of the result: the operand's shape with the
    /// axis removed.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).without_axis(self.axis)
    }
}

impl<Data: Tensorial> Operation<Data> for SumAlong {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).sum_along(self.axis)
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        smallvec![Some(gradient.broadcast_along(self.axis, operand))]
    }
}
