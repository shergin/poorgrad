use crate::{Tensorial, ValueId};

use super::Operation;

/// The transposition of a value.
///
/// Transposition is linear and self-adjoint in shape: the gradient of the
/// operand is the transposed incoming gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Transpose {
    pub(crate) operand: ValueId,
}

impl Transpose {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }
}

impl<Data: Tensorial> Operation<Data> for Transpose {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].transposed()
    }

    fn backward(&self, _values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        gradients[operand] = gradients[operand].clone() + gradient.transposed();
    }
}
