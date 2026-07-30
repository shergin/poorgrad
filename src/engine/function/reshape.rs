use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::Operation;

/// A reshape of a value to a new shape of the same volume.
///
/// Reshaping preserves logical row-major order, so it is a bijection on
/// elements: the gradient of the operand is the incoming gradient reshaped
/// back to the operand's own shape.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Reshape {
    pub(crate) operand: ValueId,
    pub(crate) shape: Shape,
}

impl Reshape {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }

    /// Infers the result shape: the requested shape, which must match the
    /// operand's volume.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        let operand = shape_of(self.operand);
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
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].reshape(self.shape.clone())
    }

    fn backward(&self, values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        let operand_shape = values[operand].shape();
        gradients[operand] = gradients[operand].clone() + gradient.reshape(operand_shape);
    }
}
