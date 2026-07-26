use crate::{Differentiable, ValueId};

use super::Operation;

/// A learnable parameter: a leaf that `Network::updated` replaces with a
/// freshly updated payload on each training step.
///
/// It behaves exactly like `Leaf` during runs: `forward` reproduces the
/// payload and `backward` is a no-op. The distinction exists so a gradient
/// step knows which leaves are trainable and which are plain data.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Parameter<Data>(pub(crate) Data);

impl<Data> Parameter<Data> {
    /// Calls `visitor` with each operand link; a parameter has none.
    pub(crate) fn visit_operands(&self, _visitor: impl FnMut(ValueId)) {}
}

impl<Data: Differentiable> Operation<Data> for Parameter<Data> {
    fn forward(&self, _values: &[Data]) -> Data {
        self.0.clone()
    }

    fn backward(&self, _values: &[Data], _gradient: &Data, _gradients: &mut [Data]) {}
}
