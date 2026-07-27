use std::ptr;

use static_assertions::assert_impl_all;

use super::{Differentiable, Field, Tape, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Evaluation<'static, f64>: Send, Sync);

/// The materialized payloads of one forward run over a `Network`.
///
/// It is per-run state: a run never mutates the network, so any number of
/// evaluations can exist and be produced concurrently. Unlike a bare
/// `Field`, an evaluation is generation-pinned — its numbers are a
/// function of one generation's parameter payloads — so it borrows that
/// generation exactly and exposes no cross-generation algebra.
#[derive(Debug)]
pub struct Evaluation<'network, Data> {
    tape: &'network Tape<Data>,
    values: Field<Data>,
}

impl<'network, Data: Differentiable> Evaluation<'network, Data> {
    pub(crate) fn new(tape: &'network Tape<Data>, values: Vec<Data>) -> Self {
        Self {
            tape,
            values: Field::new(tape.lineage().clone(), values),
        }
    }

    /// Returns the computed payload of `value`.
    ///
    /// It is the shared read-back accessor of every position-indexed
    /// buffer: evaluations, gradients, and fields all answer `of(value)`.
    ///
    /// # Panics
    /// Panics if `value` belongs to a different network or was allocated
    /// after this evaluation ran.
    pub fn of(&self, value: Value<'_, Data>) -> &Data {
        assert!(
            ptr::eq(self.tape, value.tape()),
            "value belongs to a different network"
        );
        self.values
            .as_slice()
            .get(value.id().index())
            .expect("value was allocated after this evaluation ran")
    }

    pub(crate) fn tape(&self) -> &'network Tape<Data> {
        self.tape
    }

    pub(crate) fn values(&self) -> &[Data] {
        self.values.as_slice()
    }
}
