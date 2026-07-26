use crate::{Elementary, Shape, ValueId};

use super::Operation;

/// The hyperbolic tangent of a value.
///
/// The derivative is `1 - tanh(x)^2`: one minus the square of the node's
/// own output, so `backward` reuses the computed output instead of
/// recomputing the transcendental.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tanh {
    pub(crate) operand: ValueId,
}

impl Tanh {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }

    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        shape_of(self.operand)
    }
}

impl<Data: Elementary> Operation<Data> for Tanh {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].tanh()
    }

    fn backward(&self, _values: &[Data], output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        let derivative = output.one_like() - output.clone() * output.clone();
        gradients[operand] = gradients[operand].clone() + gradient.clone() * derivative;
    }
}
