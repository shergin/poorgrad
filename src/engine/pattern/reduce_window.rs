use smallvec::SmallVec;

use crate::function::Function;
use crate::{Differentiable, Tensorial};

use super::candidates::Candidate;
use super::pattern::Pattern;
use super::view::View;

/// A matched max-pool window group: the recorded left fold of
/// `maximum` over the window lanes, rooted at the facade's squeeze
/// reshape. Emission raises the group to `stablehlo.reduce_window`
/// over the rank-4 source; a fusing forward run replaces it with
/// one direct window walk in the same lane order, bit-identical to
/// the recorded fold.
///
/// Matching is structural and provenance-blind: any recording of the
/// canonical `max_pool` composition matches — two square unfolds, the
/// lane permute and reshape, a left-associated `maximum` fold in lane
/// order, and the trailing squeeze. A balanced fold tree, a permuted
/// lane order, or an omitted squeeze is a documented false negative.
#[derive(Debug, Clone)]
pub(crate) struct ReduceWindow {
    /// The rank-4 `[batch, channels, height, width]` source.
    pub(crate) source: usize,
    /// The square window extent along both spatial axes.
    pub(crate) size: usize,
    /// The window step along both spatial axes.
    pub(crate) stride: usize,
}

impl ReduceWindow {
    /// Returns the slots the fused call reads past the root's operand
    /// links; liveness must keep them alive until the call.
    pub(crate) fn reads(&self) -> [usize; 1] {
        [self.source]
    }

    /// Computes the fused call over the already-evaluated `values`:
    /// one direct window walk from the source, the lane views between
    /// them never materialized.
    pub(crate) fn apply<Data: Tensorial>(&self, values: &[Data]) -> Data {
        values[self.source].max_pooled(self.size, self.stride)
    }
}

/// Returns the lane start of `index` if it is a `narrow(4, start, 1)`
/// of the one shared lanes node, learning that node on first sight.
fn lane_start<Data: Differentiable>(
    index: usize,
    view: &View<Data>,
    lanes: &mut Option<usize>,
) -> Option<usize> {
    let Some(Function::Narrow(narrow)) = view.function(index) else {
        return None;
    };
    if narrow.axis != 4 || narrow.len != 1 {
        return None;
    }
    let operand = view.sole_operand(index);
    match lanes {
        Some(lanes) if *lanes != operand => None,
        Some(_) => Some(narrow.start),
        None => {
            *lanes = Some(operand);
            Some(narrow.start)
        }
    }
}

/// Matches the canonical max-pool window chain rooted at `index` —
/// the squeeze `Reshape` of a left-associated `maximum` fold over
/// `reshape(permute(unfold(unfold(x))))` lanes — and returns the
/// candidate group. Interiors are collected by walking the formula,
/// not via single-consumer chains: the lanes node fans out into one
/// `narrow` per window element, and `Catalog::collect` checks the
/// keep-set and sharing closure.
pub(crate) fn match_at<Data: Differentiable>(index: usize, view: &View<Data>) -> Option<Candidate> {
    // The root is the facade squeeze: a rank-4 reshape of a rank-5
    // value whose lane axis has folded down to extent 1.
    let Some(Function::Reshape(reshape)) = view.function(index) else {
        return None;
    };
    if reshape.shape.rank() != 4 {
        return None;
    }
    let folded = view.sole_operand(index);
    let folded_axes = view.shape(folded).axes();
    if folded_axes.len() != 5 || folded_axes[4] != 1 || reshape.shape.axes() != &folded_axes[..4] {
        return None;
    }

    // Descend the left-associated fold: every right operand is one
    // lane narrow, and the chain bottoms out at the lane-0 narrow.
    let mut interiors: SmallVec<[usize; 8]> = SmallVec::new();
    let mut lanes: Option<usize> = None;
    let mut descending_starts: Vec<usize> = Vec::new();
    let mut current = folded;
    while let Some(Function::Maximum(_)) = view.function(current) {
        interiors.push(current);
        let start = lane_start(view.operand(current, 1), view, &mut lanes)?;
        descending_starts.push(start);
        interiors.push(view.operand(current, 1));
        current = view.operand(current, 0);
    }
    if lane_start(current, view, &mut lanes)? != 0 {
        return None;
    }
    interiors.push(current);
    let lanes = lanes.expect("the bottom narrow named the lanes");
    // Row-major lane order: the fold reads lanes `0, 1, ..` bottom-up,
    // so the descent sees the tail-first suffix.
    let count = descending_starts.len() + 1;
    if !descending_starts.iter().copied().eq((1..count).rev()) {
        return None;
    }

    // The lanes head: the merging reshape, the lane permute, and the
    // two square unfolds over a rank-4 source.
    let Some(Function::Reshape(lanes_reshape)) = view.function(lanes) else {
        return None;
    };
    let permuted = view.sole_operand(lanes);
    let Some(Function::Permute(permute)) = view.function(permuted) else {
        return None;
    };
    if permute.order.as_slice() != [0, 1, 2, 4, 3, 5] {
        return None;
    }
    let windows_w = view.sole_operand(permuted);
    let Some(Function::Unfold(unfold_w)) = view.function(windows_w) else {
        return None;
    };
    if unfold_w.axis != 4 || unfold_w.dilation != 1 {
        return None;
    }
    let windows_h = view.sole_operand(windows_w);
    let Some(Function::Unfold(unfold_h)) = view.function(windows_h) else {
        return None;
    };
    if unfold_h.axis != 2
        || unfold_h.dilation != 1
        || unfold_h.size != unfold_w.size
        || unfold_h.step != unfold_w.step
    {
        return None;
    }
    let source = view.sole_operand(windows_h);
    let source_axes = view.shape(source).axes();
    if source_axes.len() != 4 {
        return None;
    }

    let size = unfold_h.size;
    let stride = unfold_h.step;
    if count != size * size {
        return None;
    }
    let (batch, channels) = (source_axes[0], source_axes[1]);
    let out_height = (source_axes[2] - size) / stride + 1;
    let out_width = (source_axes[3] - size) / stride + 1;
    if lanes_reshape.shape.axes() != [batch, channels, out_height, out_width, size * size] {
        return None;
    }
    if reshape.shape.axes() != [batch, channels, out_height, out_width] {
        return None;
    }

    interiors.extend_from_slice(&[lanes, permuted, windows_w, windows_h]);
    Some(Candidate {
        pattern: Pattern::ReduceWindow(ReduceWindow {
            source,
            size,
            stride,
        }),
        root: index,
        interiors,
        named: SmallVec::new(),
    })
}

#[cfg(test)]
#[path = "tests/reduce_window_tests.rs"]
mod tests;
