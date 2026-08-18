use smallvec::SmallVec;

use super::ValueId;

/// The operand links of one recorded node, in the operation's positional
/// order.
///
/// It is one entry of the tape's operands column: the function column
/// holds what a node computes, this column holds which earlier nodes it
/// reads. The links are stored inline up to the arity of every current
/// operation, so scanning the column stays free of pointer chasing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Operands(SmallVec<[ValueId; 2]>);

impl Operands {
    /// Creates the empty operand list of a source node.
    pub(crate) fn none() -> Self {
        Self(SmallVec::new())
    }

    /// Creates the operand list holding `links` in order.
    pub(crate) fn from_slice(links: &[ValueId]) -> Self {
        Self(SmallVec::from_slice(links))
    }

    /// Returns the links as a positional slice.
    pub(crate) fn as_slice(&self) -> &[ValueId] {
        &self.0
    }
}
