use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

use cow_vec::CowVec;

use super::{Differentiable, Value, ValueId, ValueInner};

/// The lock-guarded storage shared by a `Network` and all of its `Value`
/// proxies.
///
/// It owns the roster of allocated `ValueInner`s. The roster is append-only:
/// existing nodes are never mutated or removed, so a `ValueId` handed out
/// once stays valid for the lifetime of the core.
#[derive(Debug)]
pub(crate) struct NetworkCore<Data> {
    values: RwLock<CowVec<ValueInner<Data>>>,
}

impl<Data: Differentiable> NetworkCore<Data> {
    fn new(values: CowVec<ValueInner<Data>>) -> Self {
        Self {
            values: RwLock::new(values),
        }
    }

    /// Appends `inner` to the roster and returns its handle.
    pub(crate) fn alloc(&self, inner: ValueInner<Data>) -> ValueId {
        let mut values = self.write();
        values.push(inner);
        ValueId(values.len() - 1)
    }

    /// Runs `reader` over the `ValueInner` behind `id` while holding the
    /// read lock.
    ///
    /// # Panics
    /// Panics if `id` is not allocated in this core.
    pub(crate) fn with_inner<Output>(
        &self,
        id: ValueId,
        reader: impl FnOnce(&ValueInner<Data>) -> Output,
    ) -> Output {
        let values = self.read();
        let inner = values
            .get(id.0)
            .expect("`ValueId` is out of bounds for its network");
        reader(inner)
    }

    /// Creates an independent copy of the core in O(1).
    ///
    /// The copy shares the underlying arena but keeps its own roster, so
    /// later allocations on either core never affect the other.
    fn fork(&self) -> Self {
        Self::new(self.read().clone())
    }

    fn len(&self) -> usize {
        self.read().len()
    }

    fn read(&self) -> RwLockReadGuard<'_, CowVec<ValueInner<Data>>> {
        self.values.read().expect("network lock is poisoned")
    }

    fn write(&self) -> RwLockWriteGuard<'_, CowVec<ValueInner<Data>>> {
        self.values.write().expect("network lock is poisoned")
    }
}

/// A memory management bag owning every value of one computation graph.
///
/// It stores the allocated value nodes in a `CowVec` and hands out cheap
/// `Value` proxies pointing into it. Allocation is append-only and goes
/// through a handle shared with every proxy, so an expression such as
/// `let x = v1 + v2;` grows this same network without cloning it and without
/// disturbing anything allocated before. Cloning a `Network` forks it in
/// O(1): the clone shares the underlying arena but keeps an independent
/// roster, so later allocations on either side never affect the other. The
/// network is `Send + Sync` whenever `Data` is.
#[derive(Debug)]
pub struct Network<Data> {
    core: Arc<NetworkCore<Data>>,
}

impl<Data: Differentiable> Network<Data> {
    /// Creates an empty `Network`.
    pub fn new() -> Self {
        Self {
            core: Arc::new(NetworkCore::new(CowVec::new())),
        }
    }

    /// Allocates a leaf (a network input or a learnable parameter) and
    /// returns a proxy to it.
    pub fn leaf(&self, data: Data) -> Value<Data> {
        let id = self.core.alloc(ValueInner::leaf(data));
        Value::bind(Arc::clone(&self.core), id)
    }

    /// Returns this network's own proxy for the node behind `value`, or
    /// `None` if no node with that position is allocated here.
    ///
    /// Proxies stay attached to the network that created them, so a proxy
    /// made before a fork resolves against the original network; `rebind`
    /// produces the equivalent proxy for this network. It checks only the
    /// node's position, so `value` is expected to come from this network or
    /// from a network sharing its history.
    pub fn rebind(&self, value: &Value<Data>) -> Option<Value<Data>> {
        let id = value.id();
        if id.0 >= self.len() {
            return None;
        }
        Some(Value::bind(Arc::clone(&self.core), id))
    }

    /// Returns the number of allocated values.
    pub fn len(&self) -> usize {
        self.core.len()
    }

    /// Returns `true` if it holds no values.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Evaluates every node in dependency order, returning a value buffer
    /// indexed by allocation order.
    pub fn forward(&self) -> Vec<Data> {
        todo!("evaluate each node from its inputs into a fresh value buffer")
    }

    /// Propagates gradients backward from `output`, returning a gradient
    /// buffer indexed by allocation order.
    ///
    /// It seeds the output gradient with `one_like` and accumulates into a
    /// buffer initialized with `zero_like`, leaving the network untouched.
    /// That separation of per-run state from the shared structure is what
    /// lets many threads differentiate the same network at once.
    pub fn backward(&self, _values: &[Data], _output: &Value<Data>) -> Vec<Data> {
        todo!("reverse-mode accumulation into a fresh gradient buffer")
    }
}

impl<Data: Differentiable> Clone for Network<Data> {
    /// Forks the network in O(1).
    ///
    /// The fork shares the underlying arena but keeps an independent roster:
    /// later allocations on either network never affect the other, while
    /// every `ValueId` allocated before the fork stays valid in both.
    fn clone(&self) -> Self {
        Self {
            core: Arc::new(self.core.fork()),
        }
    }
}

impl<Data: Differentiable> Default for Network<Data> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[path = "tests/network_tests.rs"]
mod tests;
