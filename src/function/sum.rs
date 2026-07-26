use crate::{Tensorial, ValueId};

use super::Operation;

/// The sum of every value in a payload, reduced to a single value.
///
/// Summation and broadcasting are adjoint: the gradient of the operand is
/// the incoming single-value gradient spread back across the operand's
/// shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sum {
    pub(crate) operand: ValueId,
}

impl Sum {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }
}

impl<Data: Tensorial> Operation<Data> for Sum {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].sum()
    }

    fn backward(&self, values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        gradients[operand] = gradients[operand].clone() + gradient.broadcast_like(&values[operand]);
    }
}
