use std::collections::HashMap;
use std::sync::Arc;

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Tensorial};

use super::plan::WindowProduct;
use super::{
    Designation, Field, Function, Gradients, Misbinding, Structure, Value, ValueRef, Witness,
};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Run<f64>: Send, Sync);

/// The materialized payloads of one forward run over a `Network`.
///
/// A run is immutable, per-run state: the graph structure frozen
/// at the start of the run and the payloads that run produced. It carries
/// no borrow of the network — kinship is checked through the same
/// [`Witness`] a [`Field`] uses — so runs outlive the generation
/// that produced them and can be stashed, moved, or differentiated
/// concurrently without pinning a `Network`. Later recordings and
/// parameter updates do not change its values or the operations
/// differentiated by [`Run::backward`].
/// The producer-specific shape of one run: which slots answer reads,
/// and whether `backward` may differentiate it.
///
/// Every forward path yields the same `Run`, but the four producers
/// leave it in genuinely different states; the posture names that
/// state as one explicit sum, so an impossible combination — remat
/// recipes on a run that refuses `backward` — cannot be represented.
/// Masked slots hold shape-correct zero placeholders that reads must
/// never answer with, so `of` and `backward` consult the posture
/// first.
#[derive(Debug)]
pub(crate) enum Posture {
    /// Full interpreter run: every slot is genuine.
    Complete,
    /// Target-sliced interpreter run: the ancestor closure of the
    /// declared targets was computed; every slot outside it holds a
    /// placeholder.
    Sliced { computed: Vec<bool> },
    /// Forward-only plan run: only the keep-set answers reads, and
    /// `backward` is refused — the liveness pass freed the buffers
    /// it would need.
    Observed { readable: Arc<Vec<bool>> },
    /// Training plan run: the keep-set answers reads, and `backward`
    /// rematerializes the dropped slots through the fused recipes.
    Training {
        readable: Arc<Vec<bool>>,
        dropped: Arc<Vec<bool>>,
        fused_patches: Arc<HashMap<usize, WindowProduct>>,
    },
}

impl Posture {
    /// Returns the mask of slots that answer reads, `None` for a
    /// complete run where every slot does.
    fn mask(&self) -> Option<&[bool]> {
        match self {
            Posture::Complete => None,
            Posture::Sliced { computed } => Some(computed),
            Posture::Observed { readable } | Posture::Training { readable, .. } => Some(readable),
        }
    }

    /// Returns whether runs of this posture may differentiate: only a
    /// forward-only plan run refuses, because its liveness pass freed
    /// the forward values the derivative rules read.
    fn differentiable(&self) -> bool {
        !matches!(self, Posture::Observed { .. })
    }
}

#[derive(Debug)]
pub struct Run<Data> {
    /// Frozen node columns for this run: functions, operands, and the
    /// shapes inferred at record time.
    structure: Structure<Data>,
    field: Field<Data>,
    posture: Posture,
}

impl<Data: Differentiable> Run<Data> {
    pub(crate) fn new(
        structure: Structure<Data>,
        witness: Witness,
        values: Vec<Data>,
        posture: Posture,
    ) -> Self {
        debug_assert_eq!(structure.len(), values.len());
        if let Some(mask) = posture.mask() {
            debug_assert_eq!(structure.len(), mask.len());
        }
        Self {
            structure,
            field: Field::new(witness, values),
            posture,
        }
    }

    /// Returns whether this run computed the slot at `index` as a
    /// readable value, as opposed to leaving a placeholder there.
    fn computed(&self, index: usize) -> bool {
        match self.posture.mask() {
            Some(mask) => mask[index],
            None => true,
        }
    }

    /// Locates `value` in this run's slots through the field's one
    /// kinship probe ([`Field::locate`]), formatting misbindings into
    /// this run's own panic messages.
    fn locate(&self, value: impl ValueRef<Data>) -> usize {
        let designation = value.designation();
        let subject = match &designation {
            Designation::Bound { .. } => "value",
            Designation::Named(_) => "symbol",
        };
        match self.field.locate(designation) {
            Ok(index) => index,
            Err(Misbinding::ForeignOrigin) => {
                panic!("{subject} belongs to a different network lineage")
            }
            Err(Misbinding::DivergentBranch) => {
                panic!("{subject} belongs to a divergent fork of the network")
            }
            Err(Misbinding::OutOfCoverage) => {
                panic!("{subject} was allocated after this run")
            }
        }
    }

    /// Returns the computed payload of `value`, named by a bound
    /// [`Value`] or a detached [`Symbol`](crate::Symbol).
    ///
    /// It is the shared read-back accessor of every position-indexed
    /// buffer: runs, gradients, and fields all answer `of(value)`.
    ///
    /// # Panics
    /// Panics if `value` belongs to a different lineage or a divergent
    /// fork, was allocated after this run, or was skipped by
    /// a target-sliced run (see
    /// [`Network::forward_for`](crate::Network::forward_for)): a
    /// placeholder must never read as a result.
    pub fn of(&self, value: impl ValueRef<Data>) -> &Data {
        let index = self.locate(value);
        assert!(
            self.computed(index),
            "value was not computed by this target-sliced run; add it to the targets"
        );
        &self.field.payloads()[index]
    }

    /// Returns the run's computed values as a field, for the displays
    /// that plot a whole pass rather than read one value out of it.
    #[cfg(feature = "evcxr")]
    pub(crate) fn field(&self) -> &Field<Data> {
        &self.field
    }

    /// Assembles a [`Gradients`] field from recorded gradient values:
    /// each `(parameter, gradient)` pair copies the gradient node's
    /// payload from this run into the parameter's slot, with
    /// zeros everywhere else — the field [`Run::backward`]
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
    /// Panics as [`Run::of`] panics for either half of a pair,
    /// if a pair's first value is not a parameter, or if a gradient's
    /// payload shape differs from its parameter's recorded shape.
    pub fn recorded_gradients<'value>(
        &self,
        pairs: impl IntoIterator<Item = (Value<'value, Data>, Value<'value, Data>)>,
    ) -> Gradients<Data>
    where
        Data: 'value,
    {
        let values = self.field.payloads();
        let mut gradients: Vec<Data> = values.iter().map(|value| value.zero_like()).collect();
        for (parameter, gradient) in pairs {
            let index = self.locate(parameter);
            assert!(
                matches!(
                    self.structure.functions.get(index),
                    Some(Function::Parameter(_))
                ),
                "recorded gradients pair each parameter with its gradient; the first \
                 value of a pair is not a parameter"
            );
            let payload = self.of(gradient).clone();
            assert_eq!(
                payload.shape(),
                parameter.shape(),
                "recorded gradient shape does not match its parameter's"
            );
            gradients[index] = payload;
        }
        Field::new(self.field.witness().clone(), gradients)
    }
}

impl<Data: Tensorial> Run<Data> {
    /// Propagates gradients backward from `output`, returning the
    /// gradient of `output` with respect to every value of this run.
    ///
    /// The target must be a scalar (rank 0): a gradient is always of one
    /// chosen scalar, so a non-scalar value is reduced explicitly with
    /// `sum` before differentiation, never summed implicitly.
    ///
    /// It seeds the output gradient with `one_like` and accumulates into
    /// a fresh buffer initialized with `zero_like`, scanning this
    /// run's own structure in reverse allocation order. Only the
    /// ancestors of `output` execute their derivative rules: every other
    /// value's gradient is exactly zero, and expressions the target does
    /// not depend on — including singular ones such as a division by
    /// zero, even when the target uses them purely as a shape or index
    /// reference — cannot disturb the result. The run holds no
    /// network borrow, so any number of threads can differentiate one
    /// shared run for their own targets at once. Values recorded
    /// after this run are absent from the result, exactly as
    /// they are absent from `of`.
    ///
    /// # Panics
    /// Panics if `output` is not a scalar, belongs to a different
    /// lineage or divergent fork, was allocated after this run
    /// ran, or was skipped by a target-sliced run.
    pub fn backward(&self, output: impl ValueRef<Data>) -> Gradients<Data> {
        let output_index = self.locate(output);
        let values = self.field.payloads();
        // A sliced run evaluates the whole ancestor closure of its
        // targets, so any computed output has every operand its
        // backward needs.
        assert!(
            self.computed(output_index),
            "value was not computed by this target-sliced run; add it to the targets"
        );
        assert!(
            self.posture.differentiable(),
            "this run came from a forward-only plan, whose liveness pass freed \
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
            self.structure
                .shapes
                .get(output_index)
                .expect("shapes cover the run")
                .rank(),
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
            let function = self
                .structure
                .functions
                .get(index)
                .expect("snapshot cannot shrink");
            let links = self
                .structure
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
        Field::new(self.field.witness().clone(), gradients)
    }

    /// Returns the genuine value at `index`, rematerializing a dropped
    /// slot by recursively re-running its rule over resolved operands —
    /// bit-identical to the forward, since the rules are pure and
    /// sources are never dropped.
    fn resolved(&self, index: usize, recomputed: &mut HashMap<usize, Data>) -> Data {
        let Posture::Training {
            dropped,
            fused_patches,
            ..
        } = &self.posture
        else {
            return self.field.payloads()[index].clone();
        };
        if !dropped[index] {
            return self.field.payloads()[index].clone();
        }
        if let Some(hit) = recomputed.get(&index) {
            return hit.clone();
        }
        // A fused chain's patches rebuild with one fast fill from the
        // source; the interior views beneath never resolve at all.
        if let Some(recipe) = fused_patches.get(&index) {
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
            .structure
            .operands
            .get(index)
            .expect("snapshot cannot shrink")
            .as_slice();
        let operand_values: SmallVec<[Data; 2]> = links
            .iter()
            .map(|link| self.resolved(link.index(), recomputed))
            .collect();
        let operands: SmallVec<[&Data; 2]> = operand_values.iter().collect();
        let function = self
            .structure
            .functions
            .get(index)
            .expect("snapshot cannot shrink");
        // Sources are never dropped, so the parameter and input arms
        // are unreachable here and their slots can stay empty.
        let value = function.forward(&operands, &[], &[]);
        recomputed.insert(index, value.clone());
        value
    }
}

#[cfg(test)]
#[path = "tests/run_tests.rs"]
mod tests;
