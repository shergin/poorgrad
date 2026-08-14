use cow_vec::CowVec;

use crate::Shape;
use crate::engine::{Function, ValueId};

use super::Operands;

/// The origin-invariant node columns of a tape.
///
/// Three equal-length columns describe every recorded node: what it
/// computes, which earlier nodes it reads, and the shape inferred when
/// it was recorded. Runs replay functions and operands; shapes are the
/// cold column used at record time and by structure consumers (plans,
/// zero placeholders). Parameter and input payloads live outside this
/// type — they turn over per generation and per run — and live identity
/// (tip claim, branch chain) lives in [`super::Identity`].
///
/// Fork, update, and snapshot share one set of columns for a family's
/// whole lifetime: `update` replaces the parameter store but never
/// touches structure.
#[derive(Debug, Clone)]
pub(crate) struct Structure<Data> {
    pub(crate) functions: CowVec<Function<Data>>,
    pub(crate) operands: CowVec<Operands>,
    pub(crate) shapes: CowVec<Shape>,
}

impl<Data> Structure<Data> {
    /// Creates empty columns.
    pub(crate) fn new() -> Self {
        Self {
            functions: CowVec::new(),
            operands: CowVec::new(),
            shapes: CowVec::new(),
        }
    }

    /// Returns the number of recorded nodes.
    pub(crate) fn len(&self) -> usize {
        self.functions.len()
    }

    /// Appends one node and returns its handle.
    ///
    /// The three columns stay equal length; callers supply a shape that
    /// has already been inferred and validated against the operands.
    pub(crate) fn push(
        &mut self,
        function: Function<Data>,
        operands: Operands,
        shape: Shape,
    ) -> ValueId {
        self.functions.push(function);
        self.operands.push(operands);
        self.shapes.push(shape);
        debug_assert_eq!(self.functions.len(), self.operands.len());
        debug_assert_eq!(self.functions.len(), self.shapes.len());
        ValueId(self.functions.len() - 1)
    }

    /// Rebuilds the columns into private arenas sized to the live nodes.
    ///
    /// A plain clone keeps sharing any sibling-polluted arena;
    /// compaction copies only the live entries so the sibling garbage
    /// can drop when every sharer is gone.
    pub(crate) fn compacted(&self) -> Self
    where
        Data: Clone,
    {
        Self {
            functions: self.functions.to_vec().into(),
            operands: self.operands.to_vec().into(),
            shapes: self.shapes.to_vec().into(),
        }
    }
}
