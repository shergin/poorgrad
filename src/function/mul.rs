use crate::{Differentiable, ValueId};

use super::Operation;

/// The product of two values.
///
/// The derivative with respect to each operand is the other operand, so
/// `backward` scales the incoming gradient by the opposite side's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Mul {
    pub(crate) left: ValueId,
    pub(crate) right: ValueId,
}

impl Mul {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.left);
        visitor(self.right);
    }
}

impl<Data: Differentiable> Operation<Data> for Mul {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.left.index()].clone() * values[self.right.index()].clone()
    }

    fn backward(&self, values: &[Data], gradient: &Data, gradients: &mut [Data]) {
        let left = self.left.index();
        let right = self.right.index();
        gradients[left] = gradients[left].clone() + gradient.clone() * values[right].clone();
        gradients[right] = gradients[right].clone() + gradient.clone() * values[left].clone();
    }
}
