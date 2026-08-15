use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// A value placed at `start ..` along one axis inside zeros of
/// `full_extent`.
///
/// The forward is [`Tensorial::pad`]; the gradient of the operand is the
/// incoming gradient with the window read back out, which is what
/// [`Tensorial::narrow`] computes — the two operations are adjoint, each
/// the other's gradient rule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Pad {
    pub(crate) axis: usize,
    pub(crate) start: usize,
    pub(crate) full_extent: usize,
}

impl Pad {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads its operand for shape only, which a placeholder answers.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the result shape: the operand's shape with `axis` widened
    /// to `full_extent`, requiring the operand's window to lie within it.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            self.axis < operand.rank(),
            "pad axis {} is out of rank for {operand}",
            self.axis
        );
        let len = operand.axes()[self.axis];
        let end = self
            .start
            .checked_add(len)
            .expect("pad window end overflows `usize`");
        assert!(
            end <= self.full_extent,
            "pad window {}..{end} exceeds the full extent {}",
            self.start,
            self.full_extent
        );
        Shape::new(operand.axes().iter().enumerate().map(|(index, &extent)| {
            if index == self.axis {
                self.full_extent
            } else {
                extent
            }
        }))
    }
}

impl<Data: Tensorial> Operation<Data> for Pad {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).pad(self.axis, self.start, self.full_extent)
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        let len = operand.shape().axes()[self.axis];
        smallvec![Some(gradient.narrow(self.axis, self.start, len))]
    }
}
