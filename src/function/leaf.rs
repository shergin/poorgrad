use crate::{Differentiable, Shape, ValueId};

use super::Operation;

/// A leaf node: a network input or a learnable parameter.
///
/// It holds its payload as the operation's parameter. `forward` reproduces
/// the payload and `backward` is a no-op, since leaves are where gradients
/// stop and get read out.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Leaf<Data>(pub(crate) Data);

impl<Data> Leaf<Data> {
    /// Calls `visitor` with each operand link; a leaf has none.
    pub(crate) fn visit_operands(&self, _visitor: impl FnMut(ValueId)) {}
}

impl<Data: Differentiable> Leaf<Data> {
    /// Infers the shape of the result: the payload's own shape.
    pub(crate) fn inferred_shape(&self) -> Shape {
        self.0.shape()
    }
}

impl<Data: Differentiable> Operation<Data> for Leaf<Data> {
    fn forward(&self, _values: &[Data]) -> Data {
        self.0.clone()
    }

    fn backward(
        &self,
        _values: &[Data],
        _output: &Data,
        _gradient: &Data,
        _gradients: &mut [Data],
    ) {
    }
}
