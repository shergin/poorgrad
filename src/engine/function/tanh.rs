use crate::engine::ValueId;
use crate::{Elementary, Shape};

use super::{Operation, unary};

/// The hyperbolic tangent of a value.
///
/// The derivative is `1 - tanh(x)^2`: one minus the square of the node's
/// own output, so `backward` reuses the computed output instead of
/// recomputing the transcendental.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Tanh;

impl Tanh {
    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl<Data: Elementary> Operation<Data> for Tanh {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        values[unary(operands).index()].tanh()
    }

    fn backward(
        &self,
        operands: &[ValueId],
        _values: &[Data],
        output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let operand = unary(operands).index();
        let derivative = output.one_like() - output.clone() * output.clone();
        gradients[operand] = gradients[operand].clone() + gradient.clone() * derivative;
    }
}
