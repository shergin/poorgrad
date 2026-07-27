use crate::{Differentiable, Shape, ValueId};

/// A learnable parameter: a leaf that `Network::updated` replaces with a
/// freshly updated payload on each training step.
///
/// It behaves exactly like `Leaf` during runs: supplied rather than
/// computed, with no gradients routed back. The distinction exists so a
/// gradient step knows which leaves are trainable and which are plain
/// data.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Parameter<Data>(pub(crate) Data);

impl<Data> Parameter<Data> {
    /// Calls `visitor` with each operand link; a parameter has none.
    pub(crate) fn visit_operands(&self, _visitor: impl FnMut(ValueId)) {}
}

impl<Data: Differentiable> Parameter<Data> {
    /// Infers the shape of the result: the payload's own shape.
    pub(crate) fn inferred_shape(&self) -> Shape {
        self.0.shape()
    }
}
