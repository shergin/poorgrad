use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::{Operation, unary};

/// The sum of every value in a payload, reduced to a single value.
///
/// Summation and broadcasting are adjoint: the gradient of the operand is
/// the incoming single-value gradient spread back across the operand's
/// shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sum;

impl Sum {
    /// Infers the shape of the result: a rank-0 single value.
    pub(crate) fn infer_shape(&self, _operands: &[Shape]) -> Shape {
        Shape::scalar()
    }
}

impl<Data: Tensorial> Operation<Data> for Sum {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        values[unary(operands).index()].sum()
    }

    fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let operand = unary(operands).index();
        gradients[operand] = gradients[operand].clone() + gradient.broadcast_like(&values[operand]);
    }
}
