use std::sync::Arc;

use cow_vec::CowVec;
use smallvec::SmallVec;
use static_assertions::assert_impl_all;

use crate::{Differentiable, Shape, Tensorial};

use super::{Evaluation, Function, Lineage, Network, Operands, Segment, Symbol};

// Compile-time thread-safety contract; the anchor rationale is documented
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
    /// Per node, the slots this node is the last consumer of and which
    /// may therefore be freed right after it computes. Empty for
    /// training plans, which retain everything `backward` might read.
    frees: Vec<SmallVec<[usize; 2]>>,
    /// Whether evaluations of this plan may differentiate: training
    /// plans retain every closure value, forward-only plans free
    /// buffers that `backward` would need.
    training: bool,
}

impl<Data: Differentiable> Plan<Data> {
    /// Compiles the plan for `network`: reachability from the roots,
    /// the readable set, and — for forward-only plans — the free
    /// lists.
    fn new(network: &Network<Data>, targets: &[Symbol], keep: &[Symbol], training: bool) -> Self {
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

        // Forward-only liveness: a slot may be freed by its highest
        // consumer inside the closure, unless the caller may read it.
        let mut frees: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); length];
        if !training {
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
                if let Some(consumer) = last_consumer[slot] {
                    frees[consumer].push(slot);
                }
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
            frees,
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

    /// Renders the plan's decisions: one line per evaluated node with
    /// its operation, shape, and liveness, then the summary — node and
    /// readable counts, and the static live-volume story (in elements;
    /// constants and placeholders count as zero, so the figures are the
    /// plan's own accounting, not allocator truth).
    pub fn describe(&self) -> String {
        use std::fmt::Write;

        let mut lines = String::new();
        let mut freed_after: Vec<Option<usize>> = vec![None; self.len()];
        for (index, frees) in self.frees.iter().enumerate() {
            for &slot in frees {
                freed_after[slot] = Some(index);
            }
        }

        let mut live: usize = 0;
        let mut peak: usize = 0;
        let mut peak_at: usize = 0;
        let mut total: usize = 0;
        let mut evaluated: usize = 0;
        for index in 0..self.len() {
            if !self.wanted[index] {
                continue;
            }
            evaluated += 1;
            let volume = self.shapes[index].volume();
            total += volume;
            live += volume;
            if live > peak {
                peak = live;
                peak_at = index;
            }
            let function = self.functions.get(index).expect("plan columns are fixed");
            let liveness = if self.readable[index] {
                "kept".to_string()
            } else {
                match freed_after[index] {
                    Some(consumer) => format!("freed after {consumer}"),
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
            for &slot in &self.frees[index] {
                live -= self.shapes[slot].volume();
            }
        }
        let mode = if self.training {
            "training (retain all)"
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
        writeln!(
            lines,
            "live volume: peak {peak} elements at node {peak_at}, retain-all {total}",
        )
        .expect("writing to a string cannot fail");
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
        Plan::new(self, &targets, &keep, false)
    }

    /// Compiles a training [`Plan`] whose evaluations differentiate
    /// `loss` exactly: every closure value is retained for `backward`,
    /// and `loss` joins the readable set alongside `keep`.
    ///
    /// # Panics
    /// Panics if `loss` or a keep does not resolve in this generation.
    pub fn compile_training(
        &self,
        loss: Symbol,
        keep: impl IntoIterator<Item = Symbol>,
    ) -> Plan<Data> {
        let keep: Vec<Symbol> = keep.into_iter().collect();
        Plan::new(self, &[loss], &keep, true)
    }
}

#[cfg(test)]
#[path = "tests/plan_tests.rs"]
mod tests;
