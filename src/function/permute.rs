use smallvec::{SmallVec, smallvec};

use crate::{Shape, Tensorial};

use super::{Cotangents, Operation, Reads, unary};

/// A permutation of a value's axes: axis `i` of the result takes axis
/// `order[i]` of the operand.
///
/// The gradient of the operand is the incoming gradient reordered by the
/// inverse permutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Permute {
    pub(crate) order: SmallVec<[usize; 4]>,
}

impl Permute {
    /// Returns the arity: one operand.
    pub(crate) fn arity(&self) -> usize {
        1
    }

    /// Returns the read set of the derivative rule below.
    /// It reads no payloads: the cotangent permutes back.
    pub(crate) fn reads(&self) -> Reads {
        Reads::NOTHING
    }

    /// Infers the result shape: the operand's axes reordered by `order`,
    /// which must be a permutation of the operand's axes.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let operand = unary(operands);
        assert_eq!(
            self.order.len(),
            operand.rank(),
            "permute order must cover every axis of {operand}"
        );
        let mut seen = vec![false; operand.rank()];
        for &axis in &self.order {
            assert!(
                axis < operand.rank(),
                "permute axis {axis} is out of rank for {operand}"
            );
            assert!(
                !std::mem::replace(&mut seen[axis], true),
                "permute order repeats axis {axis}"
            );
        }
        Shape::new(self.order.iter().map(|&axis| operand.axes()[axis]))
    }

    /// Returns the inverse permutation: the order that undoes `self.order`.
    fn inverse(&self) -> SmallVec<[usize; 4]> {
        let mut inverse: SmallVec<[usize; 4]> =
            std::iter::repeat_n(0usize, self.order.len()).collect();
        for (position, &axis) in self.order.iter().enumerate() {
            inverse[axis] = position;
        }
        inverse
    }
}

impl<Data: Tensorial> Operation<Data> for Permute {
    fn forward(&self, operands: &[&Data]) -> Data {
        unary(operands).permute(&self.order)
    }

    fn backward(&self, _operands: &[&Data], _output: &Data, gradient: &Data) -> Cotangents<Data> {
        smallvec![Some(gradient.permute(&self.inverse()))]
    }
}
