use smallvec::smallvec;

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Retention, unary};

/// The sliding windows of a value along one axis: the axis becomes a
/// `(count, size)` pair where window `w` starts at `w * step` and takes
/// every `dilation`-th element.
///
/// The forward is the [`Tensorial::unfold`] strided view; the gradient
/// of the operand is the incoming gradient with every window
/// contribution summed back onto its source position, which is what
/// [`Tensorial::fold`] computes — the two operations are adjoint. The
/// `dilation` slot ships ahead of any consumer so dilated convolution
/// later is a formula parameter, not a variant change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Unfold {
    pub(crate) axis: usize,
    pub(crate) size: usize,
    pub(crate) step: usize,
    pub(crate) dilation: usize,
}

impl Unfold {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the retention of the derivative rule below.
    /// It reads its operand for shape only, which a placeholder answers.
    pub(crate) fn retains(&self) -> Retention {
        Retention::NOTHING
    }

    /// Infers the result shape: `axis` replaced by the `(count, size)`
    /// window pair, requiring nonzero parameters and a dilated window
    /// span within the axis extent.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert!(
            self.axis < operand.rank(),
            "unfold axis {} is out of rank for {operand}",
            self.axis
        );
        assert!(
            self.size > 0,
            "unfold windows must hold at least one element"
        );
        assert!(self.step > 0, "unfold step must be positive");
        assert!(self.dilation > 0, "unfold dilation must be positive");
        let extent = operand.axes()[self.axis];
        let span = self
            .dilation
            .checked_mul(self.size - 1)
            .and_then(|reach| reach.checked_add(1))
            .expect("unfold window span overflows `usize`");
        assert!(
            span <= extent,
            "unfold window span {span} exceeds axis {} extent {extent}",
            self.axis
        );
        let count = (extent - span) / self.step + 1;
        let mut unfolded: Vec<usize> = operand.axes().to_vec();
        unfolded[self.axis] = count;
        unfolded.insert(self.axis + 1, self.size);
        Shape::new(unfolded)
    }
}

impl<Data: Tensorial> Operation<Data> for Unfold {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).unfold(self.axis, self.size, self.step, self.dilation)
    }

    fn backward(&self, operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        let &operand = unary(operands);
        let extent = operand.shape().axes()[self.axis];
        smallvec![Some(gradient.fold(
            self.axis,
            self.size,
            self.step,
            self.dilation,
            extent
        ))]
    }
}
