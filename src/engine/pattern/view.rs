use smallvec::SmallVec;

use crate::function::Function;
use crate::graph::Structure;
use crate::{Differentiable, Shape};

/// The columns and derived sets every matcher reads.
///
/// It is borrowed from `Plan::new` after `wanted` and `readable` are
/// closed and before liveness. Matchers do not mutate it and do not
/// see claims: `Catalog::collect` owns claiming and posture.
///
/// Two legality checks live here, one per action.
/// [`View::interior_ok`] is fuse legality: a home-fusing matcher walks
/// it per step, because fusing replaces the chain with placeholders,
/// so every swallowed node must be private. [`View::closed`] is raise
/// legality: the group must be airtight at the emit boundary, checked
/// once by `Catalog::collect` for every candidate.
pub(crate) struct View<'plan, Data> {
    structure: &'plan Structure<Data>,
    wanted: &'plan [bool],
    readable: &'plan [bool],
    /// Per-link consumer counts over wanted nodes; `Mul(x, x)` counts
    /// `x` twice.
    consumers: Vec<usize>,
    /// The wanted nodes that list this node as an operand, one entry
    /// per link. Used by [`View::closed`].
    consumer_of: Vec<SmallVec<[usize; 2]>>,
}

impl<'plan, Data: Differentiable> View<'plan, Data> {
    pub(crate) fn new(
        structure: &'plan Structure<Data>,
        wanted: &'plan [bool],
        readable: &'plan [bool],
    ) -> Self {
        let length = structure.len();
        let mut consumers = vec![0usize; length];
        let mut consumer_of: Vec<SmallVec<[usize; 2]>> = vec![SmallVec::new(); length];
        for (index, &wanted_node) in wanted.iter().enumerate() {
            if !wanted_node {
                continue;
            }
            let links = structure
                .operands
                .get(index)
                .expect("plan columns are fixed");
            for link in links.as_slice() {
                consumers[link.index()] += 1;
                consumer_of[link.index()].push(index);
            }
        }
        Self {
            structure,
            wanted,
            readable,
            consumers,
            consumer_of,
        }
    }

    pub(crate) fn len(&self) -> usize {
        self.structure.len()
    }

    pub(crate) fn wanted(&self, index: usize) -> bool {
        self.wanted[index]
    }

    /// Returns whether `index` can be an interior of a home-fused
    /// chain: wanted, outside the keep-set, and consumed by exactly
    /// one operand link. The single-consumer bound also keeps the
    /// fused call's own reads out of the skipped chain (`matmul(x, x)`
    /// must not swallow `x`), which [`View::closed`] cannot see.
    /// Raise-only matchers collect their spelling freely and rely on
    /// `closed` alone: their interiors still execute at home.
    pub(crate) fn interior_ok(&self, index: usize) -> bool {
        self.wanted[index] && !self.readable[index] && self.consumers[index] == 1
    }

    /// Returns whether the union of the root, the unnamed interiors,
    /// and the named results is a closed subgraph for keep-set and
    /// sharing — raise legality, checked once per candidate: unnamed
    /// interiors are wanted and not readable, named results are wanted
    /// (readable allowed) and disjoint from the interiors, and every
    /// wanted consumer of an interior or named node lies inside the
    /// set. The root may have consumers outside.
    pub(crate) fn closed(&self, root: usize, interiors: &[usize], named: &[usize]) -> bool {
        let mut member = vec![false; self.len()];
        member[root] = true;
        for &node in interiors {
            if !self.wanted[node] || self.readable[node] {
                return false;
            }
            member[node] = true;
        }
        for &node in named {
            if !self.wanted[node] || member[node] {
                return false;
            }
            member[node] = true;
        }
        for &node in interiors.iter().chain(named) {
            for &consumer in &self.consumer_of[node] {
                if !member[consumer] {
                    return false;
                }
            }
        }
        true
    }

    pub(crate) fn function(&self, index: usize) -> Option<&'plan Function<Data>> {
        self.structure.functions.get(index)
    }

    pub(crate) fn shape(&self, index: usize) -> &'plan Shape {
        &self.structure.shapes[index]
    }

    pub(crate) fn operand(&self, index: usize, position: usize) -> usize {
        self.structure
            .operands
            .get(index)
            .expect("plan columns are fixed")
            .as_slice()[position]
            .index()
    }

    pub(crate) fn sole_operand(&self, index: usize) -> usize {
        self.operand(index, 0)
    }
}
