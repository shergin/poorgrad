use crate::engine::ValueId;
use crate::{Elementary, Shape};

use super::Operation;

/// The natural logarithm of a value.
///
/// The derivative of `ln(x)` is `1 / x`, so `backward` divides the
/// incoming gradient by the operand's value. Gradients inherit the
/// payload's logarithm and division semantics outside the positive
/// domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Ln {
    pub(crate) operand: ValueId,
}

impl Ln {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        shape_of(self.operand)
    }
}

impl<Data: Elementary> Operation<Data> for Ln {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].ln()
    }

    fn backward(&self, values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        gradients[operand] =
            gradients[operand].clone() + gradient.clone() / values[operand].clone();
    }
}
