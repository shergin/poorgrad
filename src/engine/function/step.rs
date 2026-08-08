use smallvec::smallvec;

use crate::{Elementary, Shape};

use super::{Cotangents, Operation, Retention, binary};

/// The elementwise 0/1 indicator of `operand >= threshold`: the
/// Heaviside step, with operands `[operand, threshold]` and ties
/// answering one, exactly as [`Elementary::step`] defines.
///
/// It is the derivative mask of the `maximum` family recorded as a
/// node, which is what closes the op set under differentiation: the
/// relu and maximum rules speak `step`, so their recorded gradients
/// need it as an opcode. Both cotangents are `None` — the function is
/// locally constant almost everywhere, so no gradient flows through
/// it, and second derivatives of relu networks stay exact zeros
/// rather than `NaN`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Step;

impl Step {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Returns the retention of the derivative rule below.
    /// It reads nothing: both cotangents are structural `None`s.
    pub(crate) fn retains(&self) -> Retention {
        Retention::NOTHING
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (operand, threshold) = binary(operands);
        assert_eq!(operand, threshold, "step requires operands of equal shapes");
        operand.clone()
    }
}

impl<Data: Elementary> Operation<Data> for Step {
    fn forward(&self, operands: &[&Data]) -> Data {
        let (&operand, &threshold) = binary(operands);
        operand.step(threshold)
    }

    fn backward(&self, _operands: &[&Data], _output: &Data, _gradient: &Data) -> Cotangents<Data> {
        smallvec![None, None]
    }
}

#[cfg(test)]
#[path = "tests/step_tests.rs"]
mod tests;
