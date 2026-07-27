use crate::engine::ValueId;
use crate::{Differentiable, Shape};

/// A leaf node: a network input or constant supplied at recording time.
///
/// It holds its payload directly. It is supplied rather than computed —
/// `Function`'s dispatch reproduces the payload during forward and
/// routes no gradients back, since leaves are where gradients stop and
/// get read out.
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
