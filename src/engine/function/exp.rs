use crate::engine::ValueId;
use crate::{Elementary, Shape};

use super::Operation;

/// The exponential of a value.
///
/// The derivative of `e^x` is `e^x` itself — the canonical case of
/// reusing the node's own output in `backward`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Exp {
    pub(crate) operand: ValueId,
}

impl Exp {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        shape_of(self.operand)
    }
}

impl<Data: Elementary> Operation<Data> for Exp {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].exp()
    }

    fn backward(&self, _values: &[Data], output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        gradients[operand] = gradients[operand].clone() + gradient.clone() * output.clone();
    }
}
