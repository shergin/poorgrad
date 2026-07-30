use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::Operation;

/// A window of `len` elements from `start` along one axis of a value.
///
/// The forward is an O(1) view; the gradient of the operand is the incoming
/// gradient scattered back into a zero payload of the operand's shape at the
/// window, which is what [`Tensorial::padded`] computes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Narrow {
    pub(crate) operand: ValueId,
    pub(crate) axis: usize,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl Narrow {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }

    /// Infers the result shape: the operand's shape with `axis` restricted
    /// to `len`, requiring the window to lie within that axis.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        let operand = shape_of(self.operand);
        assert!(
            self.axis < operand.rank(),
            "narrow axis {} is out of rank for {operand}",
            self.axis
        );
        let extent = operand.axes()[self.axis];
        assert!(
            self.start + self.len <= extent,
            "narrow window {}..{} exceeds axis {} extent {extent}",
            self.start,
            self.start + self.len,
            self.axis
        );
        Shape::new(
            operand
                .axes()
                .iter()
                .enumerate()
                .map(|(index, &e)| if index == self.axis { self.len } else { e }),
        )
    }
}

impl<Data: Tensorial> Operation<Data> for Narrow {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].narrowed(self.axis, self.start, self.len)
    }

    fn backward(&self, values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        let full_extent = values[operand].shape().axes()[self.axis];
        gradients[operand] =
            gradients[operand].clone() + gradient.padded(self.axis, self.start, full_extent);
    }
}
