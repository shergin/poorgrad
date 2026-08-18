use crate::{Shape, Tape, Tensorial, Value};

use super::{Module, Visitor};

/// The shape adapters chains need between stages: a fixed
/// [`Reshape`] and the batch-preserving [`Flatten`]. Both are
/// stateless modules recording a single movement node.
///
/// Reinterprets its input with a fixed shape, preserving logical
/// row-major order; the volume must not change.
pub struct Reshape {
    shape: Shape,
}

impl Reshape {
    /// Creates the adapter targeting `shape`.
    pub fn new(shape: impl Into<Shape>) -> Self {
        Self {
            shape: shape.into(),
        }
    }
}

impl<Data: Tensorial> Module<Data> for Reshape {
    fn express<'tape>(
        &self,
        _tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        input.reshape(self.shape.clone())
    }

    fn visit(&self, _visitor: &mut dyn Visitor) {}
}

/// Collapses everything after the leading batch axis:
/// `[batch, rest..]` becomes `[batch, product(rest)]`, the bridge
/// from convolutional stages to linear heads. The target shape is
/// computed from the input's recorded shape at expression time.
pub struct Flatten;

impl<Data: Tensorial> Module<Data> for Flatten {
    /// # Panics
    /// Panics if the input is rank 0: there is no batch axis to keep.
    fn express<'tape>(
        &self,
        _tape: &'tape Tape<Data>,
        input: Value<'tape, Data>,
    ) -> Value<'tape, Data> {
        let shape = input.shape();
        let axes = shape.axes();
        assert!(
            !axes.is_empty(),
            "flatten requires at least a batch axis, got {shape}"
        );
        let rest: usize = axes[1..].iter().product();
        input.reshape([axes[0], rest])
    }

    fn visit(&self, _visitor: &mut dyn Visitor) {}
}
