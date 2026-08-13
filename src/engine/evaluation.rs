use std::collections::HashMap;
use std::ptr;
use std::sync::Arc;

use cow_vec::CowVec;
use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Tensorial};

use super::plan::WindowProduct;
use super::{
    Designation, Field, Function, Gradients, Kinship, Operands, Tape, Value, ValueId, ValueRef,
};

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
    values: Field<Data>,
    /// Which slots a target-sliced run actually computed; `None` for a
    /// full run, where every slot is genuine. Skipped slots hold
    /// shape-correct zero placeholders that reads must never answer
    /// with, so `of` and `backward` check this set first.
    evaluated: Option<Vec<bool>>,
    /// Whether the forward values `backward` reads are all present or
    /// rematerializable: true for interpreter runs and training plans,
    /// false for forward-only plan runs, whose liveness pass freed
    /// buffers the derivative rules would need.
    gradients_retained: bool,
    /// Which slots a training plan dropped for rematerialization:
    /// their placeholders must never feed a derivative rule, so
    /// `backward` recomputes their values on demand.
    dropped: Option<Arc<Vec<bool>>>,
    /// The fused patch recipes by reshape slot: rematerializing those
    /// slots takes one fast fill from the source instead of replaying
    /// the view chain through the general element walk.
    fused_patches: Option<Arc<HashMap<usize, WindowProduct>>>,
}

impl<'network, Data: Differentiable> Evaluation<'network, Data> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        tape: &'network Tape<Data>,
        nodes: CowVec<Function<Data>>,
        operands: CowVec<Operands>,
        kinship: Kinship,
        values: Vec<Data>,
        evaluated: Option<Vec<bool>>,
        gradients_retained: bool,
        dropped: Option<Arc<Vec<bool>>>,
        fused_patches: Option<Arc<HashMap<usize, WindowProduct>>>,
    ) -> Self {
        debug_assert_eq!(nodes.len(), values.len());
        debug_assert_eq!(nodes.len(), operands.len());
        if let Some(evaluated) = &evaluated {
            debug_assert_eq!(nodes.len(), evaluated.len());
        }
        debug_assert!(kinship.lineage() == tape.lineage());
        Self {
            tape,
            nodes,
            operands,
            values: Field::new(kinship, values),
            evaluated,
            gradients_retained,
            dropped,
            fused_patches,
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

    /// Locates `value` in this evaluation's slots: a bound proxy
    /// proves identity by tape pointer, a symbol resolves against the
    /// borrowed tape with the checks of
    /// [`Network::resolve`](crate::Network::resolve).
    fn locate(&self, value: impl ValueRef<Data>) -> usize {
        match value.designation() {
            Designation::Bound { tape, id } => {
                assert!(
                    ptr::eq(self.tape, tape),
                    "value belongs to a different network"
                );
                id.index()
            }
            Designation::Named(symbol) => self.tape.locate(symbol).index(),
        }
    }

    /// Returns the computed payload of `value`, named by a bound
    /// [`Value`] or a detached [`Symbol`](crate::Symbol).
    ///
    /// It is the shared read-back accessor of every position-indexed
    /// buffer: evaluations, gradients, and fields all answer `of(value)`.
    ///
    /// # Panics
    /// Panics if `value` belongs to a different network, was allocated
    /// after this evaluation ran, or was skipped by a target-sliced run
    /// (see [`Network::forward_for`](crate::Network::forward_for)): a
    /// placeholder must never read as a result.
    /// Returns the run's computed values as a field, for the displays
    /// that plot a whole pass rather than read one value out of it.
    #[cfg(feature = "evcxr")]
    pub(crate) fn field(&self) -> &Field<Data> {
        &self.values
    }

    pub fn of(&self, value: impl ValueRef<Data>) -> &Data {
        let index = self.locate(value);
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

    /// Assembles a [`Gradients`] field from recorded gradient values:
    /// each `(parameter, gradient)` pair copies the gradient node's
    /// payload from this evaluation into the parameter's slot, with
    /// zeros everywhere else — the field [`Evaluation::backward`]
    /// would produce for those parameters, when the gradients were
    /// recorded by [`Network::differentiate`](crate::Network::differentiate)
    /// instead of computed by the engine.
    ///
    /// It is the bridge from recorded gradients to
    /// [`Network::update`](crate::Network::update): one forward run of
    /// a compiled `[loss, gradients...]` plan yields the update
    /// direction with no backward pass at all, and the closure suite
    /// pins the two routes bitwise.
    ///
    /// # Panics
    /// Panics as [`Evaluation::of`] panics for either half of a pair,
    /// if a pair's first value is not a parameter, or if a gradient's
    /// payload shape differs from its parameter's recorded shape.
    pub fn recorded_gradients<'value>(
        &self,
        pairs: impl IntoIterator<Item = (Value<'value, Data>, Value<'value, Data>)>,
    ) -> Gradients<Data>
    where
        Data: 'value,
    {
        let values = self.values.as_slice();
        let mut field: Vec<Data> = values.iter().map(|value| value.zero_like()).collect();
        for (parameter, gradient) in pairs {
            assert!(
                ptr::eq(self.tape, parameter.tape()),
                "value belongs to a different network"
            );
            let index = parameter.id().index();
            assert!(
                matches!(self.nodes.get(index), Some(Function::Parameter(_))),
                "recorded gradients pair each parameter with its gradient; the first \
                 value of a pair is not a parameter"
            );
            let payload = self.of(gradient).clone();
            assert_eq!(
                payload.shape(),
                parameter.shape(),
                "recorded gradient shape does not match its parameter's"
            );
            field[index] = payload;
        }
        Field::new(self.values.kinship().clone(), field)
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
    pub fn backward(&self, output: impl ValueRef<Data>) -> Gradients<Data> {
        let output_index = self.locate(output);
        let values = self.values.as_slice();
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
        assert!(
            self.gradients_retained,
            "this evaluation came from a forward-only plan, whose liveness pass freed \
             the buffers backward reads; compile with `compile_training` to differentiate"
        );
        assert_eq!(
            values[output_index].shape().rank(),
            0,
            "backward requires a scalar target; reduce it with `sum` first"
        );
        // Both views of the target's shape must agree: the payload above
        // and the recorded column here, so a payload that ignored a
        // recorded movement cannot smuggle a non-scalar target through.
        assert_eq!(
            self.tape.shape(ValueId(output_index)).rank(),
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
        // The rematerialization memo: dropped values recomputed on
        // demand live here from their first (highest-indexed) reader
        // until their own node is processed, then evict — nothing at a
        // lower index can read them again.
        let mut recomputed: HashMap<usize, Data> = HashMap::new();
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
            // Resolution is retention-guided: full values only where
            // the rule reads them, shape-correct placeholders pass
            // straight through everywhere else — so shape-only rules
            // never trigger a recompute.
            let retention = function.retains();
            let output_value = if retention.output {
                self.resolved(index, &mut recomputed)
            } else {
                values[index].clone()
            };
            let operand_values: SmallVec<[Data; 2]> = links
                .iter()
                .enumerate()
                .map(|(position, link)| {
                    if retention.operands.get(position).copied().unwrap_or(true) {
                        self.resolved(link.index(), &mut recomputed)
                    } else {
                        values[link.index()].clone()
                    }
                })
                .collect();
            let operands: SmallVec<[&Data; 2]> = operand_values.iter().collect();
            let gradient = gradients[index].clone();
            let cotangents = function.backward(&operands, &output_value, &gradient);
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
            // This node's own backward was the last possible reader of
            // its rematerialized value.
            recomputed.remove(&index);
        }
        Field::new(self.values.kinship().clone(), gradients)
    }

    /// Returns the genuine value at `index`, rematerializing a dropped
    /// slot by recursively re-running its rule over resolved operands —
    /// bit-identical to the forward, since the rules are pure and
    /// sources are never dropped.
    fn resolved(&self, index: usize, recomputed: &mut HashMap<usize, Data>) -> Data {
        let is_dropped = self.dropped.as_ref().is_some_and(|dropped| dropped[index]);
        if !is_dropped {
            return self.values.as_slice()[index].clone();
        }
        if let Some(hit) = recomputed.get(&index) {
            return hit.clone();
        }
        // A fused chain's patches rebuild with one fast fill from the
        // source; the interior views beneath never resolve at all.
        if let Some(recipe) = self
            .fused_patches
            .as_ref()
            .and_then(|patches| patches.get(&index))
        {
            let recipe = recipe.clone();
            let source = self.resolved(recipe.source, recomputed);
            let value = source.windowed_patches(
                recipe.kernel_height,
                recipe.kernel_width,
                recipe.stride,
                recipe.padding,
            );
            recomputed.insert(index, value.clone());
            return value;
        }
        let links = self
            .operands
            .get(index)
            .expect("snapshot cannot shrink")
            .as_slice();
        let operand_values: SmallVec<[Data; 2]> = links
            .iter()
            .map(|link| self.resolved(link.index(), recomputed))
            .collect();
        let operands: SmallVec<[&Data; 2]> = operand_values.iter().collect();
        let function = self.nodes.get(index).expect("snapshot cannot shrink");
        // Sources are never dropped, so the parameter and input arms
        // are unreachable here and their slots can stay empty.
        let value = function.forward(&operands, &[], &[]);
        recomputed.insert(index, value.clone());
        value
    }
}

#[cfg(test)]
#[path = "tests/evaluation_tests.rs"]
mod tests;
