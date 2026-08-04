use std::sync::Arc;

use cow_vec::CowVec;
use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Shape, Tensorial};

use super::{Evaluation, Function, Lineage, Network, Operands, Segment, Symbol};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Plan<f64>: Send, Sync);

/// The element-count threshold above which a training plan drops a
/// value for rematerialization.
///
/// It is sized to the system allocator's large-allocation class: the
/// 2026-08-03 measurements showed many small mid-run frees fragment
/// the heap and regress peak RSS, while few large frees return their
/// pages. The threshold therefore selects only the big materialized
/// copies — im2col patches, padded inputs, pooling lanes — for
/// dropping and backward-time recompute, and leaves every small value
/// in place.
const REMAT_THRESHOLD: usize = 1 << 16;

/// A compiled lowering of a recorded graph prefix: which nodes a run
/// must evaluate, which values the caller may read, and which buffers
/// may be freed the moment their last consumer has run.
///
/// A plan is the bit-exact tier of the lowering ladder: it never
/// changes what is computed — plan runs reproduce the interpreter's
/// results exactly, bit for bit — it only skips what the declared
/// targets cannot observe and releases what later nodes cannot need.
/// The tape stays the specification; the plan is a derived execution
/// schedule, and [`Plan::describe`] renders its decisions.
///
/// Plans are graph-structural: [`Network::update`] replaces parameter
/// payloads, never nodes, so one plan compiled once serves every
/// generation of a training run. [`Plan::forward`] validates kinship
/// the way `update` validates a [`Field`](crate::Field), then executes
/// the plan's own column snapshot with the network's current parameter
/// and input payloads. Recording after compilation does not disturb a
/// plan; it simply keeps serving its prefix.
#[derive(Debug, Clone)]
pub struct Plan<Data> {
    lineage: Lineage,
    chain: Arc<Vec<Segment>>,
    functions: CowVec<Function<Data>>,
    operands: CowVec<Operands>,
    /// The recorded shape of every node, captured at compile time so
    /// placeholders and the memory accounting never re-touch the tape.
    shapes: Vec<Shape>,
    /// The ancestor closure of the targets and keeps: what a run must
    /// evaluate.
    wanted: Vec<bool>,
    /// The declared observable set: targets plus keeps. Only these
    /// answer [`Evaluation::of`]; an interior value stays unreadable
    /// even when liveness happens to retain it, so the contract does
    /// not depend on the optimizer's choices.
    readable: Vec<bool>,
    /// Per node, the slots whose last forward reader this node is and
    /// which the analysis licenses for release: everything outside the
    /// keep-set and retention contract, plus the dropped
    /// (rematerialized) slots. This is the plan's memory floor.
    releases: Vec<SmallVec<[usize; 2]>>,
    /// Per node, the releases a run actually executes. Forward-only
    /// plans execute every licensed release (the measured win);
    /// training plans execute only the size-thresholded drops, whose
    /// values `backward` rematerializes on demand — many small
    /// mid-run frees measured as an RSS regression (allocator
    /// fragmentation), so small values stay put.
    frees: Vec<SmallVec<[usize; 2]>>,
    /// Which slots a training run drops for rematerialization:
    /// backward recomputes them from retained neighbors, bit-exactly.
    dropped: Arc<Vec<bool>>,
    /// Whether evaluations of this plan may differentiate: training
    /// plans keep everything `backward` reads (the retention
    /// contract), forward-only plans free those buffers too.
    training: bool,
}

impl<Data: Differentiable> Plan<Data> {
    /// Compiles the plan for `network`: reachability from the roots,
    /// the readable set, the release analysis, and the drop set for
    /// rematerialization (training plans, values of at least
    /// `remat_threshold` elements).
    fn new(
        network: &Network<Data>,
        targets: &[Symbol],
        keep: &[Symbol],
        training: bool,
        remat_threshold: usize,
    ) -> Self {
        let tape = network.tape();
        let snapshot = tape.snapshot();
        let length = snapshot.functions.len();

        let mut wanted = vec![false; length];
        let mut readable = vec![false; length];
        for symbol in targets.iter().chain(keep) {
            let index = network.resolve(*symbol).id().index();
            wanted[index] = true;
            readable[index] = true;
        }
        for index in (0..length).rev() {
            if !wanted[index] {
                continue;
            }
            let links = snapshot
                .operands
                .get(index)
                .expect("snapshot cannot shrink");
            for link in links.as_slice() {
                wanted[link.index()] = true;
            }
        }

        let shapes: Vec<Shape> = (0..length)
            .map(|index| tape.shape(super::ValueId(index)))
            .collect();

        // Liveness: a slot may be freed by its highest consumer inside
        // the closure once nothing later can read its value — neither
        // the caller (the readable set) nor, in a training plan, any
        // derivative rule. Retention names exactly the payloads whose
        // values `backward` reads; shape-only readers are safe because
        // freed slots keep shape-correct placeholders.
        let mut required = readable.clone();
        if training {
            for index in 0..length {
                if !wanted[index] {
                    continue;
                }
                let function = snapshot
                    .functions
                    .get(index)
                    .expect("snapshot cannot shrink");
                let retention = function.retains();
                if retention.output {
                    required[index] = true;
                }
                let links = snapshot
                    .operands
                    .get(index)
                    .expect("snapshot cannot shrink");
                for (position, link) in links.as_slice().iter().enumerate() {
                    if retention.operands[position] {
                        required[link.index()] = true;
                    }
                }
            }
        }
        // The drop set: large values a training run rematerializes.
        // Sources are never dropped (recompute recursion bottoms out on
        // them), readables answer the caller, and small values stay put
        // — the fragmentation lesson.
        let mut dropped = vec![false; length];
        if training {
            for index in 0..length {
                if !wanted[index] || readable[index] {
                    continue;
                }
                let function = snapshot
                    .functions
                    .get(index)
                    .expect("snapshot cannot shrink");
                if function.is_source() {
                    continue;
                }
                if shapes[index].volume() >= remat_threshold {
                    dropped[index] = true;
                }
            }
        }

        let mut releases: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); length];
        let mut frees: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); length];
        let mut last_consumer: Vec<Option<usize>> = vec![None; length];
        for index in 0..length {
            if !wanted[index] {
                continue;
            }
            let links = snapshot
                .operands
                .get(index)
                .expect("snapshot cannot shrink");
            for link in links.as_slice() {
                last_consumer[link.index()] = Some(index);
            }
        }
        for slot in 0..length {
            if !wanted[slot] || readable[slot] {
                continue;
            }
            let releasable = !required[slot] || dropped[slot];
            if !releasable {
                continue;
            }
            let Some(consumer) = last_consumer[slot] else {
                continue;
            };
            releases[consumer].push(slot);
            // Forward-only plans execute every licensed release (bulk,
            // occasional runs measured a clear RSS win). Training plans
            // execute only the size-thresholded drops: per-step small
            // frees measured an RSS regression (macOS, 2026-08-03:
            // MNIST 743 MiB retain-all vs 1.16-1.23 GiB freeing), while
            // dropped large buffers land in the allocator's
            // page-returning class and are rematerialized by backward.
            if !training || dropped[slot] {
                frees[consumer].push(slot);
            }
        }

        Self {
            lineage: tape.lineage(),
            chain: snapshot.chain,
            functions: snapshot.functions,
            operands: snapshot.operands,
            shapes,
            wanted,
            readable,
            releases,
            frees,
            dropped: Arc::new(dropped),
            training,
        }
    }

    /// Returns the number of nodes in the plan's graph prefix.
    pub fn len(&self) -> usize {
        self.functions.len()
    }

    /// Returns `true` if the plan covers no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Simulates a run's live volume under `releases`, returning the
    /// peak and where it occurs, plus the retain-all total.
    fn live_story(&self, releases: &[SmallVec<[usize; 2]>]) -> (usize, usize, usize) {
        let mut live: usize = 0;
        let mut peak: usize = 0;
        let mut peak_at: usize = 0;
        let mut total: usize = 0;
        for index in 0..self.len() {
            if !self.wanted[index] {
                continue;
            }
            let volume = self.shapes[index].volume();
            total += volume;
            live += volume;
            if live > peak {
                peak = live;
                peak_at = index;
            }
            for &slot in &releases[index] {
                live -= self.shapes[slot].volume();
            }
        }
        (peak, peak_at, total)
    }

    /// Renders the plan's decisions: one line per evaluated node with
    /// its operation, shape, and liveness, then the summary — node and
    /// readable counts, and the static live-volume story (in elements;
    /// constants and placeholders count as zero, so the figures are the
    /// plan's own accounting, not allocator truth). Training plans
    /// report their retention *floor* — what the analysis could
    /// release — alongside what a run actually holds.
    pub fn describe(&self) -> String {
        use std::fmt::Write;

        let mut lines = String::new();
        let mut released_after: Vec<Option<usize>> = vec![None; self.len()];
        for (index, releases) in self.releases.iter().enumerate() {
            for &slot in releases {
                released_after[slot] = Some(index);
            }
        }
        // Executed frees match the analysis for forward-only plans;
        // training plans hold everything, so the wording distinguishes
        // what happens from what the analysis licenses.
        let release_word = if self.training {
            "releasable after"
        } else {
            "freed after"
        };

        let mut evaluated: usize = 0;
        for index in 0..self.len() {
            if !self.wanted[index] {
                continue;
            }
            evaluated += 1;
            let function = self.functions.get(index).expect("plan columns are fixed");
            let liveness = if self.readable[index] {
                "kept".to_string()
            } else if self.dropped[index] {
                match released_after[index] {
                    Some(consumer) => format!("dropped after {consumer} (remat)"),
                    None => "retained".to_string(),
                }
            } else {
                match released_after[index] {
                    Some(consumer) => format!("{release_word} {consumer}"),
                    None => "retained".to_string(),
                }
            };
            writeln!(
                lines,
                "  {index:4}  {:<14} {:<16} {liveness}",
                function.name(),
                self.shapes[index].to_string(),
            )
            .expect("writing to a string cannot fail");
        }
        let mode = if self.training {
            "training (retention analysis)"
        } else {
            "forward-only"
        };
        writeln!(
            lines,
            "plan: {mode}; {evaluated} of {} nodes evaluated, {} readable",
            self.len(),
            self.readable.iter().filter(|&&readable| readable).count(),
        )
        .expect("writing to a string cannot fail");
        let (floor, floor_at, total) = self.live_story(&self.releases);
        if self.training {
            let (executed, executed_at, _) = self.live_story(&self.frees);
            let drops = self.dropped.iter().filter(|&&dropped| dropped).count();
            let dropped_volume: usize = (0..self.len())
                .filter(|&index| self.dropped[index])
                .map(|index| self.shapes[index].volume())
                .sum();
            writeln!(
                lines,
                "live volume: retain-all {total}, remat peak {executed} elements at node \
                 {executed_at}, retention floor {floor} at node {floor_at}",
            )
            .expect("writing to a string cannot fail");
            writeln!(
                lines,
                "remat drops {drops} slots, {dropped_volume} elements, recomputed by backward",
            )
            .expect("writing to a string cannot fail");
        } else {
            writeln!(
                lines,
                "live volume: peak {floor} elements at node {floor_at}, retain-all {total}",
            )
            .expect("writing to a string cannot fail");
        }
        lines
    }
}

impl<Data: Tensorial> Plan<Data> {
    /// Runs the plan over `network`'s current generation with `feeds`
    /// bound to declared inputs for this run only, returning the
    /// evaluation of the readable values.
    ///
    /// Skipped and freed slots hold O(1) zero placeholders;
    /// [`Evaluation::of`] answers only the plan's targets and keeps,
    /// and [`Evaluation::backward`] only runs on training plans. The
    /// results of a plan run are bit-identical to the interpreter's:
    /// the plan changes what is stored, never what is computed.
    ///
    /// # Panics
    /// Panics if `network` belongs to a different lineage or a
    /// divergent fork, or as `forward_with` panics for `feeds`.
    pub fn forward<'network>(
        &self,
        network: &'network Network<Data>,
        feeds: impl IntoIterator<Item = (Symbol, Data)>,
    ) -> Evaluation<'network, Data> {
        let tape = network.tape();
        assert!(
            self.lineage == tape.lineage(),
            "plan belongs to a different network lineage"
        );
        assert!(
            tape.agrees_with_chain(&self.chain, self.len()),
            "plan belongs to a divergent fork of this network"
        );

        let mut bindings = Vec::new();
        for (symbol, payload) in feeds {
            let value = network.resolve(symbol);
            let slot = tape.input_slot(value.id()).expect("only inputs can be fed");
            assert_eq!(
                payload.shape(),
                value.shape(),
                "fed payload must match the input's recorded shape"
            );
            bindings.push((slot, payload));
        }
        let snapshot = tape.snapshot();
        let inputs = if bindings.is_empty() {
            snapshot.inputs
        } else {
            let mut overlaid = snapshot.inputs.as_ref().clone();
            for (slot, payload) in bindings {
                overlaid[slot.index()] = payload;
            }
            Arc::new(overlaid)
        };

        let mut values: Vec<Data> = Vec::with_capacity(self.len());
        for index in 0..self.len() {
            let value = if self.wanted[index] {
                let function = self.functions.get(index).expect("plan columns are fixed");
                let links = self.operands.get(index).expect("plan columns are fixed");
                let operands: SmallVec<[&Data; 2]> = links
                    .as_slice()
                    .iter()
                    .map(|link| &values[link.index()])
                    .collect();
                function.forward(&operands, snapshot.parameters.payloads(), &inputs)
            } else {
                Data::counted(self.shapes[index].clone(), 0)
            };
            values.push(value);
            // Liveness: this node was the last consumer of these
            // slots, and the caller may not read them — release now.
            for &slot in &self.frees[index] {
                values[slot] = Data::counted(self.shapes[slot].clone(), 0);
            }
        }

        Evaluation::new(
            tape,
            self.functions.clone(),
            self.operands.clone(),
            Arc::clone(&self.chain),
            values,
            Some(self.readable.clone()),
            self.training,
            Some(Arc::clone(&self.dropped)),
        )
    }
}

impl<Data: Differentiable> Network<Data> {
    /// Compiles a forward-only [`Plan`] for `targets`, with `keep`
    /// naming interior values the caller also wants readable.
    ///
    /// Forward-only plans free every non-readable buffer after its
    /// last consumer, so their evaluations refuse `backward`; compile
    /// with [`Network::compile_training`] to differentiate.
    ///
    /// # Panics
    /// Panics if a target or keep does not resolve in this generation.
    pub fn compile(
        &self,
        targets: impl IntoIterator<Item = Symbol>,
        keep: impl IntoIterator<Item = Symbol>,
    ) -> Plan<Data> {
        let targets: Vec<Symbol> = targets.into_iter().collect();
        let keep: Vec<Symbol> = keep.into_iter().collect();
        Plan::new(self, &targets, &keep, false, REMAT_THRESHOLD)
    }

    /// Compiles a training [`Plan`] whose evaluations differentiate
    /// `loss` exactly. Training runs hold every closure value — the
    /// fastest and, on the measured consumers, usually the
    /// smallest-RSS default, since the allocator recycles the uniform
    /// per-step cycle perfectly — while `describe` reports the
    /// retention floor the analysis licenses. `loss` joins the
    /// readable set alongside `keep`. See
    /// [`Network::compile_training_compact`] for the memory-leaning
    /// trade.
    ///
    /// # Panics
    /// Panics if `loss` or a keep does not resolve in this generation.
    pub fn compile_training(
        &self,
        loss: Symbol,
        keep: impl IntoIterator<Item = Symbol>,
    ) -> Plan<Data> {
        let keep: Vec<Symbol> = keep.into_iter().collect();
        Plan::new(self, &[loss], &keep, true, usize::MAX)
    }

    /// Compiles a training [`Plan`] that trades backward time for
    /// memory: large intermediates (the im2col patches, padded copies,
    /// and pooling lanes at or above the allocator's page-returning
    /// size class) are dropped right after their last forward consumer
    /// and rematerialized on demand during `backward`, bit-exactly.
    ///
    /// The trade is explicit because it does not always win: on the
    /// MNIST example it cut peak RSS 9% below retain-all for 22% more
    /// step time, while on the deeper CIFAR-10 example it cost time
    /// *and* memory (gradient cotangent buffers, not forward values,
    /// dominate there — their eviction is future work that may flip
    /// the default). Reach for it when activations, not gradients, are
    /// what does not fit.
    ///
    /// # Panics
    /// Panics if `loss` or a keep does not resolve in this generation.
    pub fn compile_training_compact(
        &self,
        loss: Symbol,
        keep: impl IntoIterator<Item = Symbol>,
    ) -> Plan<Data> {
        let keep: Vec<Symbol> = keep.into_iter().collect();
        Plan::new(self, &[loss], &keep, true, REMAT_THRESHOLD)
    }
}

#[cfg(test)]
#[path = "tests/plan_tests.rs"]
mod tests;
