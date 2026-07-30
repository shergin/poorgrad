use crate::engine::ValueId;
use crate::{Shape, Tensorial};

use super::{Operation, binary};

/// An embedding-style row gather with operands `[table, selection]`:
/// `output[i] = table[selection[i]]`, where the selection is a one-hot
/// `[count, vocab]` payload whose vocabulary is the table's first axis.
///
/// The gradient flows only to the table, `dtable[selection[i]] += grad[i]`
/// (a scatter-add that accumulates repeated rows). The selection is data,
/// not a differentiable value, so it has no gradient term at all: the
/// non-differentiability of the indices is a structural property of this
/// operation rather than a runtime flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Gather;

impl Gather {
    /// Infers the result shape `[count, ...table.shape[1..]]`, requiring the
    /// selection to be rank 2 and its vocabulary to match the table's rows.
    pub(crate) fn infer_shape(&self, operands: &[Shape]) -> Shape {
        let (table, selection) = binary(operands);
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
    fn forward(&self, operands: &[ValueId], values: &[Data]) -> Data {
        let (&table, &selection) = binary(operands);
        values[table.index()].gather(&values[selection.index()])
    }

    fn backward(
        &self,
        operands: &[ValueId],
        values: &[Data],
        _output: &Data,
        gradient: &Data,
        gradients: &mut [Data],
    ) {
        let (&table, &selection) = binary(operands);
        let table = table.index();
        let rows = values[table].shape().axes()[0];
        gradients[table] =
            gradients[table].clone() + gradient.scatter(&values[selection.index()], rows);
    }
}
