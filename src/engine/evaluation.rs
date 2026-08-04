use std::ptr;
use std::sync::Arc;

use cow_vec::CowVec;
use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Tensorial};

use super::{Field, Function, Gradients, Operands, Segment, Tape, Value};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Evaluation<'static, f64>: Send, Sync);

/// The materialized payloads of one forward run over a `Network`.
///
/// An evaluation is immutable, per-run state. It borrows the exact network
/// generation whose parameter payloads it used and retains the graph snapshot
/// captured at the start of the run. Later recordings and parameter updates do
/// not change its values or the operations differentiated by
/// [`Evaluation::backward`]. Forward and backward passes do not mutate the
/// network, so evaluations can coexist and be computed concurrently.
#[derive(Debug)]
pub struct Evaluation<'network, Data> {
    tape: &'network Tape<Data>,
    nodes: CowVec<Function<Data>>,
    operands: CowVec<Operands>,
    chain: Arc<Vec<Segment>>,
    values: Field<Data>,
    /// Which slots a target-sliced run actually computed; `None` for a
    /// full run, where every slot is genuine. Skipped slots hold
    /// shape-correct zero placeholders that reads must never answer
    /// with, so `of` and `backward` check this set first.
    evaluated: Option<Vec<bool>>,
}

impl<'network, Data: Differentiable> Evaluation<'network, Data> {
    pub(crate) fn new(
        tape: &'network Tape<Data>,
        nodes: CowVec<Function<Data>>,
        operands: CowVec<Operands>,
        chain: Arc<Vec<Segment>>,
        values: Vec<Data>,
        evaluated: Option<Vec<bool>>,
    ) -> Self {
        debug_assert_eq!(nodes.len(), values.len());
        debug_assert_eq!(nodes.len(), operands.len());
        if let Some(evaluated) = &evaluated {
            debug_assert_eq!(nodes.len(), evaluated.len());
        }
        Self {
            tape,
            nodes,
            operands,
            values: Field::new(tape.lineage(), Arc::clone(&chain), values),
            chain,
            evaluated,
        }
    }

    /// Returns whether this run computed the slot at `index`, as
    /// opposed to skipping it in a target-sliced run.
    fn computed(&self, index: usize) -> bool {
        match &self.evaluated {
            Some(evaluated) => evaluated[index],
            None => true,
        }
    }

    /// Returns the computed payload of `value`.
    ///
    /// It is the shared read-back accessor of every position-indexed
    /// buffer: evaluations, gradients, and fields all answer `of(value)`.
    ///
    /// # Panics
    /// Panics if `value` belongs to a different network, was allocated
    /// after this evaluation ran, or was skipped by a target-sliced run
    /// (see [`Network::forward_for`](crate::Network::forward_for)): a
    /// placeholder must never read as a result.
    pub fn of(&self, value: Value<'_, Data>) -> &Data {
        assert!(
            ptr::eq(self.tape, value.tape()),
            "value belongs to a different network"
        );
        let index = value.id().index();
        let payload = self
            .values
            .as_slice()
            .get(index)
            .expect("value was allocated after this evaluation ran");
        assert!(
            self.computed(index),
            "value was not evaluated by this target-sliced run; add it to the targets"
        );
        payload
    }
}

impl<'network, Data: Tensorial> Evaluation<'network, Data> {
    /// Propagates gradients backward from `output`, returning the
    /// gradient of `output` with respect to every value of this run.
    ///
    /// The target must be a scalar (rank 0): a gradient is always of one
    /// chosen scalar, so a non-scalar value is reduced explicitly with
    /// `sum` before differentiation, never summed implicitly.
    ///
    /// It seeds the output gradient with `one_like` and accumulates into
    /// a fresh buffer initialized with `zero_like`, scanning this
    /// evaluation's own tape snapshot in reverse allocation order. Only
    /// the ancestors of `output` execute their derivative rules: every
    /// other value's gradient is exactly zero, and expressions the target
    /// does not depend on — including singular ones such as a division by
    /// zero, even when the target uses them purely as a shape or index
    /// reference — cannot disturb the result. The network is not even locked,
    /// so any number of threads can differentiate one shared evaluation
    /// for their own targets at once. Values recorded after this
    /// evaluation ran are absent from the result, exactly as they are
    /// absent from `of`.
    ///
    /// # Panics
    /// Panics if `output` is not a scalar, belongs to a different
    /// network, was allocated after this evaluation ran, or was skipped
    /// by a target-sliced run.
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
        // A sliced run evaluates the whole ancestor closure of its
        // targets, so any evaluated output has every operand its
        // backward needs.
        assert!(
            self.computed(output_index),
            "value was not evaluated by this target-sliced run; add it to the targets"
        );
        assert_eq!(
            values[output_index].shape().rank(),
            0,
            "backward requires a scalar target; reduce it with `sum` first"
        );

        let mut gradients: Vec<Data> = values.iter().map(|value| value.zero_like()).collect();
        gradients[output_index] = values[output_index].one_like();
        // The single reverse scan doubles as reachability marking: every
        // consumer lives at a higher index than its operands, so when the
        // scan reaches a node it is already marked exactly when it is an
        // ancestor of the target. Skipping non-ancestors is a correctness
        // measure, not an optimization: their derivative rules must not
        // run, because a singular disconnected expression (`x / x` at
        // zero) would poison genuine gradients with NaN even through a
        // zero cotangent.
        let mut ancestors = vec![false; output_index + 1];
        ancestors[output_index] = true;
        for index in (0..=output_index).rev() {
            if !ancestors[index] {
                continue;
            }
            let function = self.nodes.get(index).expect("snapshot cannot shrink");
            let links = self
                .operands
                .get(index)
                .expect("snapshot cannot shrink")
                .as_slice();
            let operands: SmallVec<[&Data; 2]> =
                links.iter().map(|link| &values[link.index()]).collect();
            let gradient = gradients[index].clone();
            let cotangents = function.backward(&operands, &values[index], &gradient);
            debug_assert_eq!(cotangents.len(), links.len());
            // Accumulation is the multivariate chain rule: when a value
            // feeds several consumers, its gradient is the sum of the
            // cotangents arriving along every path. Only a `Some`
            // cotangent marks its operand as an ancestor: `None` declares
            // the operand data rather than a differentiable dependency
            // (a broadcast's reference, a gather's selection), so its
            // producers stay outside the scan — a singular expression
            // behind a shape-only edge must not leak NaN into genuine
            // gradients. `Some(zero)` is still an edge and still marks.
            for (&link, cotangent) in links.iter().zip(cotangents) {
                if let Some(contribution) = cotangent {
                    let slot = link.index();
                    ancestors[slot] = true;
                    gradients[slot] = gradients[slot].clone() + contribution;
                }
            }
        }
        Field::new(self.tape.lineage(), Arc::clone(&self.chain), gradients)
    }
}

#[cfg(test)]
#[path = "tests/evaluation_tests.rs"]
mod tests;
