use cow_vec::CowVec;

use crate::Shape;
use crate::engine::{Function, ValueId};

use super::Operands;

/// The node columns of a recorded graph.
///
/// Three equal-length columns describe every recorded node: what it
/// computes, which earlier nodes it reads, and the shape inferred when
/// it was recorded. Runs replay functions and operands; shapes are the
/// cold column used at record time and by structure consumers (plans,
/// zero placeholders). Parameter and input payloads live outside this
/// type: initials and defaults in the spec's stores, live state in the
/// caller's [`Parameters`](crate::Parameters).
///
/// Cloning shares the append-only column arena in O(1), which is how
/// plans and `differentiate` freeze the structure they read.
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
}
