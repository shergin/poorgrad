use std::ptr;

use static_assertions::assert_impl_all;

use super::{Differentiable, Tape, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Gradients<'static, f64>: Send, Sync);

/// The gradients of one backward run over a `Network`.
///
/// It holds the derivative of the run's output with respect to every node,
/// produced into per-run storage so the network itself stays untouched. It
/// borrows the network it was computed from and is read back with the same
/// `Value` proxies that built the graph.
#[derive(Debug)]
pub struct Gradients<'network, Data> {
    tape: &'network Tape<Data>,
    gradients: Vec<Data>,
}

impl<'network, Data: Differentiable> Gradients<'network, Data> {
    pub(crate) fn new(tape: &'network Tape<Data>, gradients: Vec<Data>) -> Self {
        Self { tape, gradients }
    }

    pub(crate) fn tape(&self) -> &'network Tape<Data> {
        self.tape
    }

    pub(crate) fn as_slice(&self) -> &[Data] {
        &self.gradients
    }

    /// Returns the gradient of the run's output with respect to `value`.
    ///
    /// # Panics
    /// Panics if `value` belongs to a different network or was allocated
    /// after the underlying evaluation ran.
    pub fn of(&self, value: Value<'_, Data>) -> &Data {
        assert!(
            ptr::eq(self.tape, value.tape()),
            "value belongs to a different network"
        );
        self.gradients
            .get(value.id().index())
            .expect("value was allocated after the underlying evaluation ran")
    }
}
