use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::{Operation, unary};

/// A reshape of a value to a new shape of the same volume.
///
/// Reshaping preserves logical row-major order, so it is a bijection on
/// elements: the gradient of the operand is the incoming gradient reshaped
/// back to the operand's own shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reshape {
    pub(crate) shape: Shape,
}

impl Reshape {
    /// Infers the result shape: the requested shape, which must match the
    /// operand's volume.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert_eq!(
            operand.volume(),
            self.shape.volume(),
            "reshape from {operand} to {} changes the number of elements",
            self.shape
        );
        self.shape.clone()
    }
}

impl<Data: Tensorial> Operation<Data> for Reshape {
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        values[unary(operands).index()].reshape(self.shape.clone())
    }

    fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let operand = unary(operands).index();
        let operand_shape = values[operand].shape();
        gradients[operand] = gradients[operand].clone() + gradient.reshape(operand_shape);
    }
}
