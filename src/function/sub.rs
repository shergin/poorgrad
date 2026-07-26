use crate::{Differentiable, Shape, ValueId};

use super::Operation;

/// The difference of two values.
///
/// The derivative with respect to the left operand is one and with
/// respect to the right operand minus one, so `backward` routes the
/// incoming gradient onward and negated respectively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Sub {
    pub(crate) left: ValueId,
    pub(crate) right: ValueId,
}

impl Sub {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.left);
        visitor(self.right);
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        let left = shape_of(self.left);
        let right = shape_of(self.right);
        assert_eq!(left, right, "subtraction requires operands of equal shapes");
        left
    }
}

impl<Data: Differentiable> Operation<Data> for Sub {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.left.index()].clone() - values[self.right.index()].clone()
    }

    fn backward(&self, _values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let left = self.left.index();
        let right = self.right.index();
        gradients[left] = gradients[left].clone() + gradient.clone();
        gradients[right] = gradients[right].clone() + -gradient.clone();
    }
}
