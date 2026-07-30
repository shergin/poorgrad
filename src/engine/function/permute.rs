use smallvec::SmallVec;

use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::Operation;

/// A permutation of a value's axes: axis `i` of the result takes axis
/// `order[i]` of the operand.
///
/// The gradient of the operand is the incoming gradient reordered by the
/// inverse permutation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Permute {
    pub(crate) operand: ValueId,
    pub(crate) order: SmallVec<[usize; 4]>,
}

impl Permute {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.operand);
    }

    /// Infers the result shape: the operand's axes reordered by `order`,
    /// which must be a permutation of the operand's axes.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        let operand = shape_of(self.operand);
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
    fn forward(&self, values: &[Data]) -> Data {
        values[self.operand.index()].permuted(&self.order)
    }

    fn backward(&self, _values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let operand = self.operand.index();
        gradients[operand] = gradients[operand].clone() + gradient.permuted(&self.inverse());
    }
}
