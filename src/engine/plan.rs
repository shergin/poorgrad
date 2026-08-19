use std::sync::Arc;

use cow_vec::CowVec;
use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Shape, Tensorial};

use crate::function::Function;
use crate::graph::{Network, Operands, Origin, Parameters, SlotStore, Structure, Symbol};

use super::pattern::{Catalog, View};
use super::{Posture, Request, Run};

// Request-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Plan<f64>: Send, Sync);

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
/// Plans are graph-structural: a plan freezes its own copy of the
/// spec — columns and input defaults — at compile time, and
/// [`Plan::forward`] takes the caller's [`Parameters`] per call, so a
/// plan never held state and there is nothing for a training step to
/// invalidate. Reopening the network and recording more does not
/// disturb a plan; it simply keeps serving its prefix.
#[derive(Debug, Clone)]
pub struct Plan<Data> {
    origin: Origin,
    /// Frozen node columns for the plan's graph prefix.
    structure: Structure<Data>,
    /// The spec's input defaults, frozen at compile time; feeds
    /// overlay them per run.
    inputs: Arc<SlotStore<Data>>,
    /// How many parameter slots the plan's prefix draws on: the
    /// coverage a [`Parameters`] value must reach.
    parameter_slots: usize,
    /// The ancestor closure of the targets and keeps: what a run must
    /// evaluate.
    wanted: Vec<bool>,
    /// The declared observable set: targets plus keeps. Only these
    /// answer [`Run::of`]; an interior value stays unreadable
    /// even when liveness happens to retain it, so the contract does
    /// not depend on the optimizer's choices.
    readable: Arc<Vec<bool>>,
    /// Per node, the slots whose last forward reader this node is and
    /// which the analysis licenses for release: everything outside the
    /// keep-set and read contract. Forward-only runs execute every
    /// licensed release; engine-backward runs execute none (many
    /// small mid-run frees measured as an RSS regression — allocator
    /// fragmentation) and report this set as their release floor.
    releases: Vec<SmallVec<[usize; 2]>>,
    /// The compiled pattern catalog: the pattern rooted at each node,
    /// if any, and the interior skip-masks its consumers read.
    catalog: Catalog,
    /// The engine-backward posture: `false` compiles forward liveness
    /// (runs refuse `backward`), `true` retains what the engine
    /// reverse scan reads.
    backward: bool,
}

impl<Data: Differentiable> Plan<Data> {
    /// Compiles the plan for `network`: reachability from the roots,
    /// the readable set, and the release analysis.
    fn new(network: &Network<Data>, roots: &[Symbol], observe: &[Symbol], backward: bool) -> Self {
        let training = backward;
        let structure = network.structure().clone();
        let length = structure.len();

        let mut wanted = vec![false; length];
        let mut readable = vec![false; length];
        for symbol in roots.iter().chain(observe) {
            let index = network.locate(*symbol).index();
            wanted[index] = true;
            readable[index] = true;
        }
        for index in (0..length).rev() {
            if !wanted[index] {
                continue;
            }
            let links = structure
                .operands
                .get(index)
                .expect("snapshot cannot shrink");
            for link in links.as_slice() {
                wanted[link.index()] = true;
            }
        }

        // Patterns: one structural match over the frozen columns, all
        // sharing the claim discipline. Home-fusing patterns are
        // gated by memory posture — fusing requires the chain to never
        // materialize, so it is a forward-only move: engine-backward
        // plans keep their exact contract unfused, the reverse scan
        // reading what the recording named.
        let view = View::new(&structure, &wanted, &readable);
        let catalog = Catalog::collect(&view, !training);

        // Liveness: a slot may be freed by its highest consumer inside
        // the closure once nothing later can read its value — neither
        // the caller (the readable set) nor, in a training plan, any
        // derivative rule. Reads names exactly the payloads whose
        // values `backward` reads; shape-only readers are safe because
        // freed slots keep shape-correct placeholders.
        let mut required = readable.clone();
        if training {
            for index in 0..length {
                if !wanted[index] {
                    continue;
                }
                let function = structure
                    .functions
                    .get(index)
                    .expect("snapshot cannot shrink");
                let reads = function.reads();
                if reads.output {
                    required[index] = true;
                }
                let links = structure
                    .operands
                    .get(index)
                    .expect("snapshot cannot shrink");
                for (position, link) in links.as_slice().iter().enumerate() {
                    if reads.operands[position] {
                        required[link.index()] = true;
                    }
                }
            }
        }
        let mut releases: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); length];
        let mut last_consumer: Vec<Option<usize>> = vec![None; length];
        for (index, &wanted_node) in wanted.iter().enumerate() {
            if !wanted_node {
                continue;
            }
            let links = structure
                .operands
                .get(index)
                .expect("snapshot cannot shrink");
            for link in links.as_slice() {
                last_consumer[link.index()] = Some(index);
            }
        }
        // A home-fusing root reads values directly, past the skipped
        // chain the operand links describe: liveness must not release
        // them before the fused call.
        catalog.pin_liveness(&mut last_consumer);
        for slot in 0..length {
            if !wanted[slot] || readable[slot] || required[slot] {
                continue;
            }
            let Some(consumer) = last_consumer[slot] else {
                continue;
            };
            // Forward-only runs execute these releases (bulk,
            // occasional runs measured a clear RSS win); engine
            // runs hold everything — per-step small frees measured
            // an RSS regression (macOS, 2026-08-03: MNIST 743 MiB
            // retain-all vs 1.16-1.23 GiB freeing), and both graded
            // consumers preferred retain over remat once the
            // recorded route existed.
            releases[consumer].push(slot);
        }

        Self {
            origin: network.origin(),
            structure,
            inputs: Arc::clone(network.inputs()),
            parameter_slots: network.parameters_len(),
            wanted,
            readable: Arc::new(readable),
            releases,
            catalog,
            backward,
        }
    }

    /// Returns the number of nodes in the plan's graph prefix.
    pub fn len(&self) -> usize {
        self.structure.len()
    }

    /// Returns `true` if the plan covers no nodes.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns whether run buffers support
    /// [`Run::backward`](crate::Run::backward): true exactly when the
    /// request asked for engine reverse mode; `describe` prints the
    /// posture.
    pub fn can_backward(&self) -> bool {
        self.backward
    }

    /// Returns the plan's function column, for plan consumers such as
    /// the StableHLO emitter — introspection siblings of `describe`.
    pub(crate) fn functions(&self) -> &CowVec<Function<Data>> {
        &self.structure.functions
    }

    /// Returns the plan's operand column, parallel to the functions.
    pub(crate) fn operands(&self) -> &CowVec<Operands> {
        &self.structure.operands
    }

    /// Returns the recorded shape of every node.
    pub(crate) fn shapes(&self) -> &CowVec<Shape> {
        &self.structure.shapes
    }

    /// Returns the ancestor closure of the targets and keeps: what a
    /// run must evaluate.
    pub(crate) fn wanted(&self) -> &[bool] {
        &self.wanted
    }

    /// Returns the declared observable set: targets plus keeps.
    pub(crate) fn readable(&self) -> &[bool] {
        &self.readable
    }

    /// Returns the compiled pattern catalog, for plan consumers such
    /// as the StableHLO emitter — introspection siblings of
    /// `describe`.
    pub(crate) fn catalog(&self) -> &Catalog {
        &self.catalog
    }

    /// Simulates a run's live volume under `releases`, returning the
    /// peak and where it occurs, plus the retain-all total.
    fn live_story(&self, releases: &[SmallVec<[usize; 2]>]) -> (usize, usize, usize) {
        let mut live: usize = 0;
        let mut peak: usize = 0;
        let mut peak_at: usize = 0;
        let mut total: usize = 0;
        for (index, slots) in releases.iter().enumerate() {
            if !self.wanted[index] || self.catalog.home_interior(index) {
                continue;
            }
            let volume = self.structure.shapes[index].volume();
            total += volume;
            live += volume;
            if live > peak {
                peak = live;
                peak_at = index;
            }
            for &slot in slots {
                // Fusion interiors were never counted live: their
                // slots hold placeholders from the start.
                if self.catalog.home_interior(slot) {
                    continue;
                }
                live -= self.structure.shapes[slot].volume();
            }
        }
        (peak, peak_at, total)
    }

    /// Returns the live volume after every evaluated node under the
    /// analysis floor: the curve whose peak [`describe`](Plan::describe)
    /// reports as one number.
    #[cfg(feature = "evcxr")]
    pub(crate) fn live_series(&self) -> Vec<f64> {
        let mut live: usize = 0;
        let mut series = Vec::new();
        for (index, slots) in self.releases.iter().enumerate() {
            if !self.wanted[index] || self.catalog.home_interior(index) {
                continue;
            }
            live += self.structure.shapes[index].volume();
            series.push(live as f64);
            for &slot in slots {
                if self.catalog.home_interior(slot) {
                    continue;
                }
                live -= self.structure.shapes[slot].volume();
            }
        }
        series
    }

    /// Renders the plan's decisions: one line per evaluated node with
    /// its operation, shape, and liveness, then the summary — node and
    /// readable counts, and the static live-volume story (in elements;
    /// constants and placeholders count as zero, so the figures are the
    /// plan's own accounting, not allocator truth). Engine-backward
    /// plans report their release *floor* — what the analysis could
    /// release — alongside the retain-all total a run actually holds.
    pub fn describe(&self) -> String {
        use std::fmt::Write;

        let mut lines = String::new();
        let mut released_after: Vec<Option<usize>> = vec![None; self.len()];
        for (index, releases) in self.releases.iter().enumerate() {
            for &slot in releases {
                released_after[slot] = Some(index);
            }
        }
        // Forward-only runs execute the analysis; engine-backward
        // runs hold everything, so the wording distinguishes what
        // happens from what the analysis licenses.
        let release_word = if self.backward {
            "releasable after"
        } else {
            "freed after"
        };

        let mut evaluated: usize = 0;
        for (index, &released) in released_after.iter().enumerate() {
            if !self.wanted[index] {
                continue;
            }
            evaluated += 1;
            let function = self
                .structure
                .functions
                .get(index)
                .expect("plan columns are fixed");
            let liveness = if self.catalog.home_interior(index) {
                "fused (window-gemm)".to_string()
            } else if self.readable[index] {
                "kept".to_string()
            } else {
                match released {
                    Some(consumer) => format!("{release_word} {consumer}"),
                    None => "retained".to_string(),
                }
            };
            writeln!(
                lines,
                "  {index:4}  {:<14} {:<16} {liveness}",
                function.name(),
                self.structure.shapes[index].to_string(),
            )
            .expect("writing to a string cannot fail");
        }
        let mode = if self.backward { "retain" } else { "forward" };
        writeln!(
            lines,
            "plan: {mode}; {evaluated} of {} nodes evaluated, {} readable",
            self.len(),
            self.readable.iter().filter(|&&readable| readable).count(),
        )
        .expect("writing to a string cannot fail");
        let groups = self.catalog.home_groups();
        if groups > 0 {
            writeln!(
                lines,
                "fused {groups} window-gemm groups, {} interior nodes skipped",
                (0..self.len())
                    .filter(|&index| self.catalog.home_interior(index))
                    .count(),
            )
            .expect("writing to a string cannot fail");
        }
        let (floor, floor_at, total) = self.live_story(&self.releases);
        if self.backward {
            writeln!(
                lines,
                "live volume: retain-all {total}, release floor {floor} at node {floor_at}",
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
    /// Runs the plan with parameter payloads read from `parameters`
    /// and `feeds` bound to declared inputs for this run only,
    /// returning a run carrying the readable values.
    ///
    /// The plan is self-contained: it executes its own frozen columns
    /// and input defaults, so no network is borrowed — the state walks
    /// in per call, which is why one plan compiled once serves every
    /// training step and every what-if.
    ///
    /// Skipped and freed slots hold O(1) zero placeholders;
    /// [`Run::of`] answers only the plan's targets and keeps,
    /// and [`Run::backward`] only runs on training plans. The
    /// results of a plan run are bit-identical to the interpreter's:
    /// the plan changes what is stored, never what is computed.
    ///
    /// # Panics
    /// Panics if `parameters` belongs to a different network or does
    /// not cover the plan's parameter slots, if a fed symbol does not
    /// name an input inside the plan's prefix, or if a fed payload's
    /// shape differs from the input's recorded shape.
    pub fn forward(
        &self,
        parameters: &Parameters<Data>,
        feeds: impl IntoIterator<Item = (Symbol, Data)>,
    ) -> Run<Data> {
        assert!(
            parameters.origin() == self.origin,
            "parameters belong to a different network"
        );
        assert!(
            parameters.len() >= self.parameter_slots,
            "parameters do not cover the plan's parameter slots; \
             carry them across a reopen with `Parameters::carried`"
        );

        let mut bindings = Vec::new();
        for (symbol, payload) in feeds {
            assert!(
                symbol.origin == self.origin,
                "symbol belongs to a different network"
            );
            let index = symbol.id.index();
            assert!(
                index < self.len(),
                "symbol is not allocated in the plan's graph prefix"
            );
            let slot = match self.structure.functions.get(index) {
                Some(Function::Input(input)) => input.0,
                _ => panic!("only inputs can be fed"),
            };
            assert_eq!(
                payload.shape(),
                self.structure.shapes[index],
                "fed payload must match the input's recorded shape"
            );
            bindings.push((slot, payload));
        }
        let inputs = if bindings.is_empty() {
            Arc::clone(&self.inputs)
        } else {
            let mut overlaid = self.inputs.as_ref().clone();
            for (slot, payload) in bindings {
                overlaid.set(slot, payload);
            }
            Arc::new(overlaid)
        };

        let mut values: Vec<Data> = Vec::with_capacity(self.len());
        for index in 0..self.len() {
            let value = if !self.wanted[index] || self.catalog.home_interior(index) {
                Data::counted(self.structure.shapes[index].clone(), 0)
            } else if let Some(group) = self.catalog.at(index).and_then(|pattern| pattern.fused()) {
                // The fused call reads its sources directly; the chain
                // between them was never materialized.
                group.apply(&values)
            } else {
                let function = self
                    .structure
                    .functions
                    .get(index)
                    .expect("plan columns are fixed");
                let links = self
                    .structure
                    .operands
                    .get(index)
                    .expect("plan columns are fixed");
                let operands: SmallVec<[&Data; 2]> = links
                    .as_slice()
                    .iter()
                    .map(|link| &values[link.index()])
                    .collect();
                let value = function.forward(&operands, parameters.payloads(), inputs.payloads());
                // The same producing-node contract check the interpreter
                // run makes: the rule's output must carry the plan's
                // recorded shape for this slot.
                debug_assert_eq!(
                    value.shape(),
                    self.structure.shapes[index],
                    "operation output shape disagrees with the recorded shape at node {index}"
                );
                value
            };
            values.push(value);
            // Liveness: this node was the last consumer of these
            // slots, and the caller may not read them — a forward-only
            // run releases now; an engine run holds everything its
            // backward reads.
            if !self.backward {
                for &slot in &self.releases[index] {
                    values[slot] = Data::counted(self.structure.shapes[slot].clone(), 0);
                }
            }
        }

        let posture = if self.backward {
            Posture::Training {
                readable: Arc::clone(&self.readable),
            }
        } else {
            Posture::Observed {
                readable: Arc::clone(&self.readable),
            }
        };
        Run::new(self.structure.clone(), self.origin, values, posture)
    }
}

impl<Data: Differentiable> Network<Data> {
    /// Compiles `request` into a [`Plan`]: the single lowering entry
    /// point, over the request's roots, observes, and engine-backward
    /// memory posture.
    ///
    /// Forward-only requests (never calling
    /// [`Request::backward`]) free every non-readable buffer
    /// after its last consumer, so their runs refuse `backward`;
    /// recorded gradient symbols compile as ordinary roots.
    ///
    /// # Panics
    /// Panics if a root or observe does not resolve in this network.
    pub fn compile(&self, request: Request) -> Plan<Data> {
        Plan::new(self, &request.roots, &request.observe, request.backward)
    }
}

#[cfg(test)]
#[path = "tests/plan_tests.rs"]
mod tests;
