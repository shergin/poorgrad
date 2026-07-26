use crate::{Differentiable, ValueId};

use super::Operation;

/// The negation of a value.
///
/// The derivative with respect to the operand is minus one, so `backward`
/// routes the negated incoming gradient to the operand.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Neg {
    pub(crate) operand: ValueId,
}

impl Neg {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }
}

impl<Data: Differentiable> Operation<Data> for Neg {
    fn forward(&self, values: &[Data]) -> Data {
        -values[self.operand.index()].clone()
    }

    fn backward(&self, _values: &[Data], gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        gradients[operand] = gradients[operand].clone() + -gradient.clone();
    }
}
