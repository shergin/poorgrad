use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Retention, unary};

/// A window of `len` elements from `start` along one axis of a value.
///
/// The forward is an O(1) view; the gradient of the operand is the incoming
/// gradient scattered back into a zero payload of the operand's shape at the
/// window, which is what [`Tensorial::pad`] computes.
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

    /// Returns the retention of the derivative rule below.
    /// It reads its operand for shape only, which a placeholder answers.
    pub(crate) fn retains(&self) -> Retention {
        Retention::NOTHING
    }

    /// Infers the result shape: the operand's shape with `axis` restricted
    /// to `len`, requiring a non-empty window lying within that axis.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            self.axis < operand.rank(),
            "narrow axis {} is out of rank for {operand}",
            self.axis
        );
        // A zero-length window would produce an empty tensor, which the
        // payload rules out by construction; rejecting it here keeps the
        // failure at recording time.
        assert!(self.len > 0, "narrow window must hold at least one element");
        let extent = operand.axes()[self.axis];
        let end = self
            .start
            .checked_add(self.len)
            .expect("narrow window end overflows `usize`");
        assert!(
            end <= extent,
            "narrow window {}..{end} exceeds axis {} extent {extent}",
            self.start,
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
        unary(operands).narrow(self.axis, self.start, self.len)
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        let full_extent = operand.shape().axes()[self.axis];
        smallvec![Some(gradient.pad(self.axis, self.start, full_extent))]
    }
}
