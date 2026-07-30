use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::Operation;

/// An embedding-style row gather: `output[i] = table[indices[i]]`, where
/// `indices` is a one-hot `[count, vocab]` selection whose vocabulary is the
/// table's first axis.
///
/// The gradient flows only to the table, `dtable[indices[i]] += grad[i]` (a
/// scatter-add that accumulates repeated rows). The selection is data, not a
/// differentiable value, so it has no gradient term at all: the
/// non-differentiability of the indices is a structural property of this
/// operation rather than a runtime flag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Gather {
    pub(crate) table: ValueId,
    pub(crate) indices: ValueId,
}

impl Gather {
    /// Calls `visitor` with each operand link.
    pub(crate) fn visit_operands(&self, mut visitor: impl FnMut(ValueId)) {
        visitor(self.table);
        visitor(self.indices);
    }

    /// Infers the result shape `[count, ...table.shape[1..]]`, requiring the
    /// selection to be rank 2 and its vocabulary to match the table's rows.
    pub(crate) fn inferred_shape(&self, shape_of: impl Fn(ValueId) -> Shape) -> Shape {
        let table = shape_of(self.table);
        let selection = shape_of(self.indices);
        assert_eq!(
            selection.rank(),
            2,
            "gather selection must be rank 2 [count, vocab], got {selection}"
        );
        assert!(
            table.rank() >= 1,
            "gather table needs at least one axis, got {table}"
        );
        assert_eq!(
            selection.axes()[1],
            table.axes()[0],
            "gather selection vocabulary {} does not match table rows {}",
            selection.axes()[1],
            table.axes()[0]
        );
        Shape::new(std::iter::once(selection.axes()[0]).chain(table.axes()[1..].iter().copied()))
    }
}

impl<Data: Tensorial> Operation<Data> for Gather {
    fn forward(&self, values: &[Data]) -> Data {
        values[self.table.index()].gather(&values[self.indices.index()])
    }

    fn backward(&self, values: &[Data], _output: &Data, gradient: &Data, gradients: &mut [Data]) {
        let table = self.table.index();
        let rows = values[table].shape().axes()[0];
        let selection = &values[self.indices.index()];
        gradients[table] = gradients[table].clone() + gradient.scatter(selection, rows);
    }
}
