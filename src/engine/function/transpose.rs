use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::{Operation, unary};

/// The transposition of a value.
///
/// Transposition is linear and self-adjoint in shape: the gradient of the
/// operand is the transposed incoming gradient.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Transpose;

impl Transpose {
    /// Infers the shape of the result: the operand's axes reversed.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            operand.rank() <= 2,
            "transpose supports rank 2 at most, got {operand}"
        );
        Shape::new(operand.axes().iter().rev().copied())
    }
}

impl<Data: Tensorial> Operation<Data> for Transpose {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        values[unary(operands).index()].transposed()
    }

    fn backward(
        &self,
        operands: &[ValueId],
        _values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let operand = unary(operands).index();
        gradients[operand] = gradients[operand].clone() + gradient.transposed();
    }
}
