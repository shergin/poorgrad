use smallvec::smallvec;

use crate::{Differentiable, Shape};

use super::{Cotangents, Operation, binary};

/// The quotient of two values, with operands `[left, right]`.
///
/// The derivative with respect to the left operand is `1 / right`; with
/// respect to the right operand it is `-left / right^2`, which equals
/// `-output / right`, so `backward` reuses the node's own output the way
/// `Tanh` does. Gradients inherit the payload's division semantics near
/// zero.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Div;

impl Div {
    /// Returns the arity: two operands.
    pub(crate) fn arity(&self) -> usize {
        2
    }

    /// Infers the shape of the result, which both operands must share.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (left, right) = binary(operands);
        assert_eq!(left, right, "division requires operands of equal shapes");
        left.clone()
    }
}

impl<Data: Differentiable> Operation<Data> for Div {
    fn forward(&self, operands: &[&Data]) -> Data {
        let (&left, &right) = binary(operands);
        left.clone() / right.clone()
    }

    fn backward(&self, operands: &[&Data], output: &Data, gradient: &Data) -> Cotangents<Data> {
        let (_, &right) = binary(operands);
        smallvec![
            Some(gradient.clone() / right.clone()),
            Some(-(gradient.clone() * output.clone() / right.clone())),
        ]
    }
}
