use std::sync::Arc;

use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Tensorial};

use super::{Field, Function, Gradients, Origin, Structure, Symbol};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Run<f64>: Send, Sync);

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
    /// Engine-backward plan run: only the keep-set answers reads,
    /// and the run retains every forward value `backward` reads.
    Training { readable: Arc<Vec<bool>> },
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

/// The materialized payloads of one forward run.
///
/// A run is immutable, per-run state: the graph structure frozen at
/// the start of the run and the payloads that run produced. It borrows
/// nothing — kinship is the same origin-and-coverage check every
/// detached carrier makes — so runs can be stashed, moved, or
/// differentiated concurrently without pinning a [`Network`](crate::Network),
/// and a reopened tape recording new nodes does not change its values
/// or the operations differentiated by [`Run::backward`].
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
        origin: Origin,
        values: Vec<Data>,
        posture: Posture,
    ) -> Self {
        debug_assert_eq!(structure.len(), values.len());
        if let Some(mask) = posture.mask() {
            debug_assert_eq!(structure.len(), mask.len());
        }
        Self {
            structure,
            field: Field::new(origin, values),
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

    /// Locates `symbol` in this run's slots.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network or was
    /// allocated after this run.
    fn locate(&self, symbol: Symbol) -> usize {
        assert!(
            symbol.origin == self.field.origin(),
            "symbol belongs to a different network"
        );
        assert!(
            symbol.id.index() < self.field.len(),
            "symbol was allocated after this run"
        );
        symbol.id.index()
    }

    /// Returns the computed payload of the value named by `symbol`.
    ///
    /// It is the shared read-back accessor of every position-indexed
    /// buffer: runs, gradients, and fields all answer `of(symbol)`.
    ///
    /// # Panics
    /// Panics if `symbol` belongs to a different network, was
    /// allocated after this run, or was skipped by a target-sliced run
    /// (see [`Network::forward_for`](crate::Network::forward_for)): a
    /// placeholder must never read as a result.
    pub fn of(&self, symbol: Symbol) -> &Data {
        let index = self.locate(symbol);
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
    /// recorded by [`Tape::differentiate`](crate::Tape::differentiate)
    /// instead of computed by the engine.
    ///
    /// It is the bridge from recorded gradients to
    /// [`Parameters::step`](crate::Parameters::step): one forward run
    /// of a compiled `[loss, gradients...]` plan yields the update
    /// direction with no backward pass at all, and the closure suite
    /// pins the two routes bitwise.
    ///
    /// # Panics
    /// Panics as [`Run::of`] panics for either half of a pair,
    /// if a pair's first symbol is not a parameter, or if a gradient's
    /// payload shape differs from its parameter's recorded shape.
    pub fn recorded_gradients(
        &self,
        pairs: impl IntoIterator<Item = (Symbol, Symbol)>,
    ) -> Gradients<Data> {
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
                 symbol of a pair is not a parameter"
            );
            let payload = self.of(gradient).clone();
            assert_eq!(
                payload.shape(),
                self.structure
                    .shapes
                    .get(index)
                    .expect("shapes cover the run")
                    .clone(),
                "recorded gradient shape does not match its parameter's"
            );
            gradients[index] = payload;
        }
        Field::new(self.field.origin(), gradients)
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
    /// reference — cannot disturb the result. The run borrows nothing,
    /// so any number of threads can differentiate one shared run for
    /// their own targets at once. Values recorded after this run are
    /// absent from the result, exactly as they are absent from `of`.
    ///
    /// # Panics
    /// Panics if `output` is not a scalar, belongs to a different
    /// network, was allocated after this run, or was skipped by a
    /// target-sliced run.
    pub fn backward(&self, output: Symbol) -> Gradients<Data> {
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
             the buffers backward reads; compile with `engine_backward` to differentiate"
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
        for index in (0..=output_index).rev() {
            if !ancestors[index] {
                continue;
            }
            let function = self
                .structure
                .functions
                .get(index)
                .expect("the freeze cannot shrink");
            let links = self
                .structure
                .operands
                .get(index)
                .expect("the freeze cannot shrink")
                .as_slice();
            // Every payload a derivative rule reads is present:
            // interpreter runs hold everything, and engine-backward
            // plan runs retain what the read contract names.
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
        Field::new(self.field.origin(), gradients)
    }
}

#[cfg(test)]
#[path = "tests/run_tests.rs"]
mod tests;
