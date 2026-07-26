use std::fmt;
use std::ops::{Add, Mul, Neg};
use std::sync::Arc;

use super::network::NetworkCore;
use super::{Differentiable, Function, ValueInner};

/// A lightweight, `Copy` handle to a value allocated in a `Network`.
///
/// It is an index into the network's storage rather than a pointer, so
/// handles are cheap to copy and carry no ownership of network memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ValueId(pub(crate) usize);

/// A cheap proxy to a value allocated in a `Network`.
///
/// It pairs a shared handle to the owning network's storage with the
/// position of its node, so it can cross threads freely and clones in O(1)
/// (it holds an `Arc`, so it cannot be `Copy`). Arithmetic operators on
/// proxies build the graph: `let x = v1 + v2;` allocates a new computed node
/// on the same network and returns a proxy to it, without cloning the
/// network. Operators are implemented for owned proxies and references
/// alike; use `&v1 + &v2` to keep the operands usable afterwards.
#[derive(Clone)]
pub struct Value<Data> {
    core: Arc<NetworkCore<Data>>,
    id: ValueId,
}

impl<Data: Differentiable> Value<Data> {
    /// Binds a proxy to the node `id` inside `core`.
    pub(crate) fn bind(core: Arc<NetworkCore<Data>>, id: ValueId) -> Self {
        Self { core, id }
    }

    /// Returns the handle of the node this proxy points to.
    pub(crate) fn id(&self) -> ValueId {
        self.id
    }

    /// Returns a clone of the `Function` that produced this value.
    // Scaffolding: only tests read it until `backward` is implemented.
    #[allow(dead_code)]
    pub(crate) fn function(&self) -> Function {
        self.core
            .with_inner(self.id, |inner| inner.function().clone())
    }

    /// Returns a clone of the leaf payload, or `None` for computed values.
    pub fn data(&self) -> Option<Data> {
        self.core.with_inner(self.id, |inner| inner.data().cloned())
    }

    /// Allocates a computed node produced by `function` on the same network
    /// and returns a proxy to it.
    fn apply(&self, function: Function) -> Self {
        let id = self.core.alloc(ValueInner::computed(function));
        Self::bind(Arc::clone(&self.core), id)
    }

    /// Panics if `other` belongs to a different network.
    fn assert_same_network(&self, other: &Self) {
        assert!(
            Arc::ptr_eq(&self.core, &other.core),
            "values belong to different networks"
        );
    }
}

/// It prints only the node position to avoid dumping the whole network.
impl<Data> fmt::Debug for Value<Data> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Value")
            .field("id", &self.id)
            .finish()
    }
}

impl<Data: Differentiable> Add for &Value<Data> {
    type Output = Value<Data>;

    fn add(self, rhs: Self) -> Value<Data> {
        self.assert_same_network(rhs);
        self.apply(Function::Add(self.id, rhs.id))
    }
}

impl<Data: Differentiable> Add for Value<Data> {
    type Output = Value<Data>;

    fn add(self, rhs: Self) -> Value<Data> {
        &self + &rhs
    }
}

impl<Data: Differentiable> Mul for &Value<Data> {
    type Output = Value<Data>;

    fn mul(self, rhs: Self) -> Value<Data> {
        self.assert_same_network(rhs);
        self.apply(Function::Mul(self.id, rhs.id))
    }
}

impl<Data: Differentiable> Mul for Value<Data> {
    type Output = Value<Data>;

    fn mul(self, rhs: Self) -> Value<Data> {
        &self * &rhs
    }
}

impl<Data: Differentiable> Neg for &Value<Data> {
    type Output = Value<Data>;

    fn neg(self) -> Value<Data> {
        self.apply(Function::Neg(self.id))
    }
}

impl<Data: Differentiable> Neg for Value<Data> {
    type Output = Value<Data>;

    fn neg(self) -> Value<Data> {
        -&self
    }
}
