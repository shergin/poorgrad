// Scaffolding: `function` is written during graph construction and is only
// read once `Network::forward`/`backward` are implemented, so `dead_code` is
// silenced until then.
#![allow(dead_code)]

use super::{Differentiable, Function};

/// The actual node of the computation graph, allocated inside a `Network`.
///
/// It couples the `Function` that produced the node with the initial payload
/// for leaves, and is reached through the cheap `Value` proxy rather than
/// held directly. Computed payloads and gradients are produced in per-run
/// buffers by `Network::forward`/`backward` and never written back into the
/// node, which is what keeps the allocated graph immutable and shareable
/// across threads.
#[derive(Debug, Clone)]
pub(crate) struct ValueInner<Data> {
    function: Function,
    /// The initial payload for `Function::Leaf` nodes; `None` for computed
    /// nodes, whose payloads are produced during the forward pass.
    data: Option<Data>,
}

impl<Data: Differentiable> ValueInner<Data> {
    /// Creates a leaf holding `data`: a network input or a learnable
    /// parameter.
    pub(crate) fn leaf(data: Data) -> Self {
        Self {
            function: Function::Leaf,
            data: Some(data),
        }
    }

    /// Creates a node whose payload will be produced by the forward pass.
    pub(crate) fn computed(function: Function) -> Self {
        Self {
            function,
            data: None,
        }
    }

    /// Returns the `Function` that produced this node.
    pub(crate) fn function(&self) -> &Function {
        &self.function
    }

    /// Returns the leaf payload, or `None` for computed nodes.
    pub(crate) fn data(&self) -> Option<&Data> {
        self.data.as_ref()
    }
}
