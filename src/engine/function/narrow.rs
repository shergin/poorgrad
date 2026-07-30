use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, unary};

/// A window of `len` elements from `start` along one axis of a value.
///
/// The forward is an O(1) view; the gradient of the operand is the incoming
/// gradient scattered back into a zero payload of the operand's shape at the
/// window, which is what [`Tensorial::padded`] computes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Narrow {
    pub(crate) axis: usize,
    pub(crate) start: usize,
    pub(crate) len: usize,
}

impl Narrow {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Infers the result shape: the operand's shape with `axis` restricted
    /// to `len`, requiring the window to lie within that axis.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
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
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).narrowed(self.axis, self.start, self.len)
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        let full_extent = operand.shape().axes()[self.axis];
        smallvec![Some(gradient.padded(self.axis, self.start, full_extent))]
    }
}
