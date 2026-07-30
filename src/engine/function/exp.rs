use crate::engine::ValueId;
use crate::{Elementary, Shape};

use super::{Operation, unary};

/// The exponential of a value.
///
/// The derivative of `e^x` is `e^x` itself — the canonical case of
/// reusing the node's own output in `backward`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Exp;

impl Exp {
    /// Infers the shape of the result: the operand's shape.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        unary(operands).clone()
    }
}

impl<Data: Elementary> Operation<Data> for Exp {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        values[unary(operands).index()].exp()
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
        gradients[operand] = gradients[operand].clone() + gradient.clone() * output.clone();
    }
}
