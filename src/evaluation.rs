use std::ptr;

use cow_vec::CowVec;
use static_assertions::assert_impl_all;

use super::{Differentiable, Field, Function, Gradients, Operation, Tape, Tensorial, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Evaluation<'static, f64>: Send, Sync);

/// The materialized payloads of one forward run over a `Network`.
///
/// It is per-run state: a run never mutates the network, so any number of
/// evaluations can exist and be produced concurrently. Unlike a bare
/// `Field`, an evaluation is generation-pinned — its numbers are a
/// function of one generation's parameter payloads — so it borrows that
/// generation exactly and exposes no cross-generation algebra. It also
/// carries the tape snapshot it was computed from, which is what lets
/// `backward` differentiate it without ever touching the network again.
#[derive(Debug)]
pub struct Evaluation<'network, Data> {
    tape: &'network Tape<Data>,
    nodes: CowVec<Function<Data>>,
    values: Field<Data>,
}

impl<'network, Data: Differentiable> Evaluation<'network, Data> {
    pub(crate) fn new(
        tape: &'network Tape<Data>,
        nodes: CowVec<Function<Data>>,
        values: Vec<Data>,
    ) -> Self {
        debug_assert_eq!(nodes.len(), values.len());
        Self {
            tape,
            nodes,
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
}

impl<'network, Data: Tensorial> Evaluation<'network, Data> {
    /// Propagates gradients backward from `output`, returning the
    /// gradient of `output` with respect to every value of this run.
    ///
    /// It seeds the output gradient with `one_like` and accumulates into
    /// a fresh buffer initialized with `zero_like`, scanning this
    /// evaluation's own tape snapshot in reverse allocation order. The
    /// network is not even locked, so any number of threads can
    /// differentiate one shared evaluation for their own targets at once.
    /// Values recorded after this evaluation ran are absent from the
    /// result, exactly as they are absent from `of`.
    ///
    /// # Panics
    /// Panics if `output` belongs to a different network or was allocated
    /// after this evaluation ran.
    pub fn backward(&self, output: Value<'_, Data>) -> Gradients<Data> {
        assert!(
            ptr::eq(self.tape, output.tape()),
            "output belongs to a different network"
        );
        let values = self.values.as_slice();
        let output_index = output.id().index();
        assert!(
            output_index < values.len(),
            "value was allocated after this evaluation ran"
        );

        let mut gradients: Vec<Data> = values.iter().map(|value| value.zero_like()).collect();
        gradients[output_index] = values[output_index].one_like();
        for index in (0..self.nodes.len()).rev() {
            let function = self.nodes.get(index).expect("snapshot cannot shrink");
            let gradient = gradients[index].clone();
            function.backward(values, &values[index], &gradient, &mut gradients);
        }
        Gradients::new(Field::new(self.tape.lineage().clone(), gradients))
    }
}
