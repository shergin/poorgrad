use super::ValueId;

/// The differentiable operation that produced a `Value`, referencing the
/// operation's inputs by `ValueId`.
///
/// It records everything the backward pass needs to apply the chain rule at a
/// single node: which operation produced the node and where its inputs live
/// in the `Network`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Function {
    /// A leaf supplied at build time: a network input or a learnable
    /// parameter.
    Leaf,
    Add(ValueId, ValueId),
    Mul(ValueId, ValueId),
    Neg(ValueId),
}
