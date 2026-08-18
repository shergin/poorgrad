use std::sync::Arc;

use cow_vec::CowVec;
use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Shape, Tensorial};

use super::{
    Compile, Function, Network, Operands, Origin, Parameters, Posture, Run, SlotStore, Structure,
    Symbol,
};

// Compile-time thread-safety contract; the anchor rationale is documented
// in `network.rs`.
assert_impl_all!(Plan<f64>: Send, Sync);

/// A matched window-GEMM fusion group: the `matmul` node computes
/// [`Tensorial::windowed_product`] directly from the source, and the
/// im2col chain between them — pads, unfolds, permute, reshape — is
/// never materialized. Backward rematerializes the chain on demand
/// through the ordinary rules, so gradients stay bit-identical.
///
/// Matching is structural and provenance-blind: any recording of the
/// canonical im2col composition fuses, whichever facade (or hand)
/// wrote it, and a keep-set node inside the chain is a fusion barrier.
#[derive(Debug, Clone)]
pub(crate) struct WindowProduct {
    /// The rank-4 `[batch, channels, height, width]` source.
    pub(crate) source: usize,
    /// The GEMM-shaped `[columns, filters]` kernel operand.
    pub(crate) kernel: usize,
    pub(crate) kernel_height: usize,
    pub(crate) kernel_width: usize,
    pub(crate) stride: usize,
    pub(crate) padding: usize,
}

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
    /// The window-GEMM fusion group rooted at each `matmul` node, if
    /// its im2col chain matched.
    fused: Vec<Option<WindowProduct>>,
    /// The interior nodes of fusion groups: skipped by runs (their
    /// slots hold placeholders) and rematerialized by backward.
    fused_interior: Vec<bool>,
    /// The engine-backward posture: `false` compiles forward liveness
    /// (runs refuse `backward`), `true` retains what the engine
    /// reverse scan reads.
    engine_backward: bool,
}

/// Scans for the canonical im2col chain feeding each `matmul` —
/// `reshape(permute(unfold(unfold(pad(pad(x)?)?))))` with the conv
/// parameterization — and returns the fusion groups plus the interior
/// mask. Matching is structural: interiors must be wanted, outside
/// the keep-set (a kept interior is a fusion barrier), and consumed
/// exactly once inside the closure.
fn match_window_products<Data: Differentiable>(
    functions: &CowVec<Function<Data>>,
    operands: &CowVec<Operands>,
    shapes: &CowVec<Shape>,
    wanted: &[bool],
    readable: &[bool],
) -> (Vec<Option<WindowProduct>>, Vec<bool>) {
    let length = functions.len();
    let mut consumers = vec![0usize; length];
    for (index, &wanted_node) in wanted.iter().enumerate() {
        if !wanted_node {
            continue;
        }
        let links = operands.get(index).expect("plan columns are fixed");
        for link in links.as_slice() {
            consumers[link.index()] += 1;
        }
    }
    let interior_ok = |index: usize| wanted[index] && !readable[index] && consumers[index] == 1;
    let sole_operand = |index: usize| {
        operands
            .get(index)
            .expect("plan columns are fixed")
            .as_slice()[0]
            .index()
    };

    let mut fused: Vec<Option<WindowProduct>> = vec![None; length];
    let mut interior = vec![false; length];
    for index in 0..length {
        if !wanted[index] {
            continue;
        }
        let Some(Function::MatMul(_)) = functions.get(index) else {
            continue;
        };
        let links = operands.get(index).expect("plan columns are fixed");
        let [lhs, kernel] = links.as_slice() else {
            continue;
        };
        let (lhs, kernel) = (lhs.index(), kernel.index());

        let Some(Function::Reshape(reshape)) = functions.get(lhs) else {
            continue;
        };
        if !interior_ok(lhs) || reshape.shape.rank() != 2 {
            continue;
        }
        let permuted = sole_operand(lhs);
        let Some(Function::Permute(permute)) = functions.get(permuted) else {
            continue;
        };
        if !interior_ok(permuted) || permute.order.as_slice() != [0, 2, 4, 1, 3, 5] {
            continue;
        }
        let windows_w = sole_operand(permuted);
        let Some(Function::Unfold(unfold_w)) = functions.get(windows_w) else {
            continue;
        };
        if !interior_ok(windows_w) || unfold_w.axis != 4 || unfold_w.dilation != 1 {
            continue;
        }
        let windows_h = sole_operand(windows_w);
        let Some(Function::Unfold(unfold_h)) = functions.get(windows_h) else {
            continue;
        };
        if !interior_ok(windows_h)
            || unfold_h.axis != 2
            || unfold_h.dilation != 1
            || unfold_h.step != unfold_w.step
        {
            continue;
        }
        let mut chain: SmallVec<[usize; 6]> =
            SmallVec::from_slice(&[lhs, permuted, windows_w, windows_h]);
        let mut source = sole_operand(windows_h);
        let mut padding = 0;
        // Symmetric zero pads fold into the fused call; anything else
        // simply leaves the pad output as the (materialized) source.
        if let Some(Function::Pad(pad_w)) = functions.get(source)
            && interior_ok(source)
            && pad_w.axis == 3
        {
            let below = sole_operand(source);
            if let Some(Function::Pad(pad_h)) = functions.get(below) {
                let base = sole_operand(below);
                let base_axes = shapes[base].axes();
                if interior_ok(below)
                    && pad_h.axis == 2
                    && base_axes.len() == 4
                    && pad_h.start == pad_w.start
                    && pad_h.full_extent == base_axes[2] + 2 * pad_h.start
                    && pad_w.full_extent == base_axes[3] + 2 * pad_w.start
                {
                    chain.push(source);
                    chain.push(below);
                    padding = pad_h.start;
                    source = base;
                }
            }
        }
        let source_axes = shapes[source].axes();
        if source_axes.len() != 4 {
            continue;
        }
        let (batch, channels) = (source_axes[0], source_axes[1]);
        let padded_height = source_axes[2] + 2 * padding;
        let padded_width = source_axes[3] + 2 * padding;
        let (kernel_height, kernel_width) = (unfold_h.size, unfold_w.size);
        let stride = unfold_h.step;
        let out_height = (padded_height - kernel_height) / stride + 1;
        let out_width = (padded_width - kernel_width) / stride + 1;
        let expected = Shape::new([
            batch * out_height * out_width,
            channels * kernel_height * kernel_width,
        ]);
        if reshape.shape != expected {
            continue;
        }
        fused[index] = Some(WindowProduct {
            source,
            kernel,
            kernel_height,
            kernel_width,
            stride,
            padding,
        });
        for &node in &chain {
            interior[node] = true;
        }
    }
    (fused, interior)
}

impl<Data: Differentiable> Plan<Data> {
    /// Compiles the plan for `network`: reachability from the roots,
    /// the readable set, and the release analysis.
    fn new(
        network: &Network<Data>,
        roots: &[Symbol],
        observe: &[Symbol],
        engine_backward: bool,
    ) -> Self {
        let training = engine_backward;
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

        // Fusion: match the canonical im2col chains. Fusing requires
        // the chain to never materialize, so it is a forward-only
        // move: engine-backward plans keep their exact contract
        // unfused — the reverse scan reads what the recording named,
        // and per-step patch re-allocation in backward measured as a
        // peak-RSS regression on the deeper consumer back when remat
        // existed.
        let (fused, fused_interior) = if !training {
            match_window_products(
                &structure.functions,
                &structure.operands,
                &structure.shapes,
                &wanted,
                &readable,
            )
        } else {
            (vec![None; length], vec![false; length])
        };

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
        // A fused matmul reads its source and kernel directly, past the
        // skipped chain the operand links describe: liveness must not
        // release them before the fused call.
        for (index, group) in fused.iter().enumerate() {
            if let Some(group) = group {
                for slot in [group.source, group.kernel] {
                    let latest = last_consumer[slot].unwrap_or(0).max(index);
                    last_consumer[slot] = Some(latest);
                }
            }
        }
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
            fused,
            fused_interior,
            engine_backward,
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
        self.engine_backward
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

    /// Returns how many window-GEMM fusion groups the plan matched.
    pub(crate) fn fusion_groups(&self) -> usize {
        self.fused.iter().flatten().count()
    }

    /// Returns the window-GEMM fusion group rooted at node `index`, if
    /// its im2col chain matched.
    pub(crate) fn fusion_group(&self, index: usize) -> Option<&WindowProduct> {
        self.fused[index].as_ref()
    }

    /// Returns which nodes are fusion-group interiors: skipped by runs
    /// and replaced wholesale by the fused call — or by the raised
    /// operation, when a plan consumer emits instead of executing.
    pub(crate) fn fused_interiors(&self) -> &[bool] {
        &self.fused_interior
    }

    /// Simulates a run's live volume under `releases`, returning the
    /// peak and where it occurs, plus the retain-all total.
    fn live_story(&self, releases: &[SmallVec<[usize; 2]>]) -> (usize, usize, usize) {
        let mut live: usize = 0;
        let mut peak: usize = 0;
        let mut peak_at: usize = 0;
        let mut total: usize = 0;
        for (index, slots) in releases.iter().enumerate() {
            if !self.wanted[index] || self.fused_interior[index] {
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
                if self.fused_interior[slot] {
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
            if !self.wanted[index] || self.fused_interior[index] {
                continue;
            }
            live += self.structure.shapes[index].volume();
            series.push(live as f64);
            for &slot in slots {
                if self.fused_interior[slot] {
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
        let release_word = if self.engine_backward {
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
            let liveness = if self.fused_interior[index] {
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
        let mode = if self.engine_backward {
            "retain"
        } else {
            "forward"
        };
        writeln!(
            lines,
            "plan: {mode}; {evaluated} of {} nodes evaluated, {} readable",
            self.len(),
            self.readable.iter().filter(|&&readable| readable).count(),
        )
        .expect("writing to a string cannot fail");
        let groups = self.fusion_groups();
        if groups > 0 {
            writeln!(
                lines,
                "fused {groups} window-gemm groups, {} interior nodes skipped",
                self.fused_interior
                    .iter()
                    .filter(|&&interior| interior)
                    .count(),
            )
            .expect("writing to a string cannot fail");
        }
        let (floor, floor_at, total) = self.live_story(&self.releases);
        if self.engine_backward {
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
            let value = if !self.wanted[index] || self.fused_interior[index] {
                Data::counted(self.structure.shapes[index].clone(), 0)
            } else if let Some(group) = &self.fused[index] {
                // The fused call reads the source and kernel directly;
                // the im2col chain between them was never materialized.
                values[group.source].windowed_product(
                    &values[group.kernel],
                    group.kernel_height,
                    group.kernel_width,
                    group.stride,
                    group.padding,
                )
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
            if !self.engine_backward {
                for &slot in &self.releases[index] {
                    values[slot] = Data::counted(self.structure.shapes[slot].clone(), 0);
                }
            }
        }

        let posture = if self.engine_backward {
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
    /// [`Compile::engine_backward`]) free every non-readable buffer
    /// after its last consumer, so their runs refuse `backward`;
    /// recorded gradient symbols compile as ordinary roots.
    ///
    /// # Panics
    /// Panics if a root or observe does not resolve in this network.
    pub fn compile(&self, request: Compile) -> Plan<Data> {
        Plan::new(
            self,
            &request.roots,
            &request.observe,
            request.engine_backward,
        )
    }
}

#[cfg(test)]
#[path = "tests/plan_tests.rs"]
mod tests;
