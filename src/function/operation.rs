use crate::Differentiable;

/// A differentiable operation: how a node computes its payload from its
/// operands, and how it routes gradients back to them.
///
/// It is implemented by each computed `Function` variant and dispatched
/// through the enum with a plain `match`, so implementations stay
/// statically sized and the trait never needs to be object safe. Leaves
/// and parameters do not implement it: they are supplied, not computed,
/// and the enum's dispatch handles them directly. `forward` and
/// `backward` read and write per-run buffers indexed by allocation
/// order, relying on the tape guarantee that operands are always
/// recorded before the nodes that use them.
pub(crate) trait Operation<Data: Differentiable> {
    /// Computes this node's payload from the values of earlier nodes.
    fn forward(&self, values: &[Data]) -> Data;

    /// Accumulates operand gradients, given this node's computed
    /// `output` payload and its own `gradient`.
    fn backward(&self, values: &[Data], output: &Data, gradient: &Data, gradients: &mut [Data]);
}
