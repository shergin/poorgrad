use std::collections::HashMap;
use std::sync::Arc;

use cow_vec::CowVec;
use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Shape, Tensorial};

use super::{
    Evaluation, Function, Lineage, Network, Operands, Segment, Symbol, ValueRef, chains_agree,
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
    /// The window-GEMM fusion group rooted at each `matmul` node, if
    /// its im2col chain matched.
    fused: Vec<Option<WindowProduct>>,
    /// The interior nodes of fusion groups: skipped by runs (their
    /// slots hold placeholders) and rematerialized by backward.
    fused_interior: Vec<bool>,
    /// The fused patch recipes keyed by their im2col reshape slot, so
    /// the backward rematerializer can rebuild a chain's patches with
    /// one fast fill instead of the general element walk.
    fused_patches: Arc<HashMap<usize, WindowProduct>>,
    /// Whether evaluations of this plan may differentiate: training
    /// plans keep everything `backward` reads (the retention
    /// contract), forward-only plans free those buffers too.
    training: bool,
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
    shapes: &[Shape],
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

        // Fusion: match the canonical im2col chains. Fusing a training
        // plan requires rematerializing the patches during backward, so
        // fusion follows the plan's memory posture: forward-only plans
        // always fuse (a pure win — the chain simply never exists), and
        // compact training plans fuse (measured strictly better), while
        // the default retain-all training plan keeps its exact contract
        // unfused — per-step patch re-allocation in backward measured
        // as a peak-RSS regression on the deeper consumer.
        let compact = remat_threshold != usize::MAX;
        let (fused, fused_interior) = if !training || compact {
            match_window_products(
                &snapshot.functions,
                &snapshot.operands,
                &shapes,
                &wanted,
                &readable,
            )
        } else {
            (vec![None; length], vec![false; length])
        };

        // The backward rematerializer rebuilds a fused chain's patches
        // with one fast fill; key the recipes by the reshape slot the
        // matmul's operand link names.
        let mut fused_patches: HashMap<usize, WindowProduct> = HashMap::new();
        for (index, group) in fused.iter().enumerate() {
            if let Some(group) = group {
                let links = snapshot
                    .operands
                    .get(index)
                    .expect("snapshot cannot shrink");
                fused_patches.insert(links.as_slice()[0].index(), group.clone());
            }
        }

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
            // Fusion interiors are never materialized, so backward must
            // always be able to rematerialize them, whatever their size.
            for index in 0..length {
                if fused_interior[index] {
                    dropped[index] = true;
                }
            }
            // Rematerialization inputs: recompute of a dropped chain
            // bottoms out at its first non-dropped ancestors, whose
            // values backward will read — the release analysis must
            // keep them, or a forced release would rebuild the chain
            // from placeholders. (Executed training frees only ever
            // release dropped slots, but the licensed set and the
            // reported floor must be honest too.)
            for index in 0..length {
                if !dropped[index] {
                    continue;
                }
                let links = snapshot
                    .operands
                    .get(index)
                    .expect("snapshot cannot shrink");
                for link in links.as_slice() {
                    if !dropped[link.index()] {
                        required[link.index()] = true;
                    }
                }
            }
        }

        let mut releases: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); length];
        let mut frees: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); length];
        let mut last_consumer: Vec<Option<usize>> = vec![None; length];
        for (index, &wanted_node) in wanted.iter().enumerate() {
            if !wanted_node {
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
            fused_patches: Arc::new(fused_patches),
            fused,
            fused_interior,
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

    /// Returns the plan's function column, for plan consumers such as
    /// the StableHLO emitter — introspection siblings of `describe`.
    pub(crate) fn functions(&self) -> &CowVec<Function<Data>> {
        &self.functions
    }

    /// Returns the plan's operand column, parallel to the functions.
    pub(crate) fn operands(&self) -> &CowVec<Operands> {
        &self.operands
    }

    /// Returns the recorded shape of every node.
    pub(crate) fn shapes(&self) -> &[Shape] {
        &self.shapes
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
            let volume = self.shapes[index].volume();
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
                live -= self.shapes[slot].volume();
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
            live += self.shapes[index].volume();
            series.push(live as f64);
            for &slot in slots {
                if self.fused_interior[slot] {
                    continue;
                }
                live -= self.shapes[slot].volume();
            }
        }
        series
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
        for (index, &released) in released_after.iter().enumerate() {
            if !self.wanted[index] {
                continue;
            }
            evaluated += 1;
            let function = self.functions.get(index).expect("plan columns are fixed");
            let liveness = if self.fused_interior[index] {
                "fused (window-gemm)".to_string()
            } else if self.readable[index] {
                "kept".to_string()
            } else if self.dropped[index] {
                match released {
                    Some(consumer) => format!("dropped after {consumer} (remat)"),
                    None => "retained".to_string(),
                }
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
    /// divergent fork, does not contain the plan's whole graph prefix,
    /// or as `forward_with` panics for `feeds`.
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
        // One snapshot serves validation and the run, so both observe
        // the same atomic tape state; chain agreement alone is not
        // containment, since a shorter sibling attributes `[0, len)`
        // to the same branches without carrying the nodes.
        let snapshot = tape.snapshot();
        assert!(
            snapshot.functions.len() >= self.len(),
            "plan covers a graph prefix this network does not contain"
        );
        assert!(
            chains_agree(&snapshot.chain, &self.chain, self.len()),
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
            let value = if !self.wanted[index] || self.fused_interior[index] {
                Data::counted(self.shapes[index].clone(), 0)
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
                let function = self.functions.get(index).expect("plan columns are fixed");
                let links = self.operands.get(index).expect("plan columns are fixed");
                let operands: SmallVec<[&Data; 2]> = links
                    .as_slice()
                    .iter()
                    .map(|link| &values[link.index()])
                    .collect();
                let value = function.forward(&operands, snapshot.parameters.payloads(), &inputs);
                // The same producing-node contract check the interpreter
                // run makes: the rule's output must carry the plan's
                // recorded shape for this slot.
                debug_assert_eq!(
                    value.shape(),
                    self.shapes[index],
                    "operation output shape disagrees with the recorded shape at node {index}"
                );
                value
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
            Some(Arc::clone(&self.fused_patches)),
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
        targets: impl IntoIterator<Item = impl ValueRef<Data>>,
        keep: impl IntoIterator<Item = Symbol>,
    ) -> Plan<Data> {
        let targets: Vec<Symbol> = targets
            .into_iter()
            .map(|target| self.named(target))
            .collect();
        let keep: Vec<Symbol> = keep.into_iter().collect();
        Plan::new(self, &targets, &keep, false, REMAT_THRESHOLD)
    }

    /// Compiles a training [`Plan`] whose evaluations differentiate
    /// `loss` exactly, holding forward values per `retention`. `loss`
    /// joins the readable set alongside `keep`.
    ///
    /// # Panics
    /// Panics if `loss` or a keep does not resolve in this generation.
    pub fn compile_training(
        &self,
        loss: impl ValueRef<Data>,
        keep: impl IntoIterator<Item = Symbol>,
        retention: Retention,
    ) -> Plan<Data> {
        let keep: Vec<Symbol> = keep.into_iter().collect();
        let threshold = match retention {
            Retention::All => usize::MAX,
            Retention::Compact => REMAT_THRESHOLD,
        };
        Plan::new(self, &[self.named(loss)], &keep, true, threshold)
    }
}

/// The forward-value retention policy of a training plan: what a run
/// holds for `backward`, chosen explicitly at the compile call site.
///
/// It replaces a fork of the compile facade — the policy is a closed
/// set of alternatives, so it is a plain `Copy` enum parameter, with
/// each variant's measured trade documented where it is chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Retention {
    /// Hold every closure value: the fastest and, on the measured
    /// consumers, usually the smallest-RSS choice, since the
    /// allocator recycles the uniform per-step cycle perfectly;
    /// `describe` reports the retention floor the analysis licenses.
    All,
    /// Trade backward time for memory: large intermediates (the
    /// im2col patches, padded copies, and pooling lanes at or above
    /// the allocator's page-returning size class) are dropped right
    /// after their last forward consumer and rematerialized on demand
    /// during `backward`, bit-exactly. The trade does not always win:
    /// on the MNIST example it cut peak RSS 9% below retain-all for
    /// 22% more step time, while on the deeper CIFAR-10 example it
    /// cost time *and* memory (gradient cotangent buffers, not
    /// forward values, dominate there — their eviction is future work
    /// that may flip the default). Reach for it when activations, not
    /// gradients, are what does not fit.
    Compact,
}

#[cfg(test)]
#[path = "tests/plan_tests.rs"]
mod tests;
