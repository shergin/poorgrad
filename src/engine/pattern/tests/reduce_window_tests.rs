use crate::function::Function;
use crate::graph::Network;
use crate::{Tape, Tensor, max_pool};

use super::super::candidates::Candidates;
use super::super::catalog::Catalog;
use super::super::pattern::Pattern;
use super::super::view::View;
use super::match_at;

/// Records a `2 x 2 / 2` max pool over a rank-4 leaf and returns the
/// network with the pooled root's index.
fn pool_network() -> (Network<Tensor<f64>>, usize) {
    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [1, 2, 4, 4],
        (0..32).map(|v| v as f64 * 0.3 - 4.0).collect::<Vec<_>>(),
    ));
    let pooled = max_pool(input, 2, 2);
    let root = pooled.symbol().id.index();
    (tape.into_network(), root)
}

/// Builds a view over the whole network with every node wanted and
/// only `readable` nodes readable.
fn full_view<'plan>(
    network: &'plan Network<Tensor<f64>>,
    wanted: &'plan [bool],
    readable: &'plan [bool],
) -> View<'plan, Tensor<f64>> {
    View::new(network.structure(), wanted, readable)
}

#[test]
fn the_pool_fold_matches_with_its_geometry() {
    let (network, root) = pool_network();
    let length = network.structure().functions.len();
    let wanted = vec![true; length];
    let readable = vec![false; length];
    let view = full_view(&network, &wanted, &readable);

    let candidate = match_at(root, &view).expect("the canonical fold matches");
    let group = match candidate.pattern {
        Pattern::ReduceWindow(group) => group,
        _ => panic!("the pool fold matches as a window reduction"),
    };
    assert_eq!(
        network.structure().shapes[group.source].axes(),
        [1, 2, 4, 4]
    );
    assert_eq!(group.size, 2);
    assert_eq!(group.stride, 2);
    // Two unfolds, the permute, the lanes reshape, four narrows, and
    // three maximums.
    assert_eq!(candidate.interiors.len(), 11);
    assert!(candidate.named.is_empty());
}

#[test]
fn a_balanced_fold_tree_does_not_match() {
    let tape = Tape::new();
    let input = tape.leaf(Tensor::new(
        [1, 2, 4, 4],
        (0..32).map(|v| v as f64 * 0.5 - 3.0).collect::<Vec<_>>(),
    ));
    let lanes = input
        .unfold(2, 2, 2, 1)
        .unfold(4, 2, 2, 1)
        .permute([0, 1, 2, 4, 3, 5])
        .reshape([1, 2, 2, 2, 4]);
    let left = lanes.narrow(4, 0, 1).maximum(lanes.narrow(4, 1, 1));
    let right = lanes.narrow(4, 2, 1).maximum(lanes.narrow(4, 3, 1));
    let pooled = left.maximum(right).squeeze(4);
    let root = pooled.symbol().id.index();
    let network = tape.into_network();

    let length = network.structure().functions.len();
    let wanted = vec![true; length];
    let readable = vec![false; length];
    let view = full_view(&network, &wanted, &readable);
    assert!(match_at(root, &view).is_none());
}

#[test]
fn a_kept_lanes_reshape_bars_the_match() {
    let (network, _root) = pool_network();
    let structure = network.structure();
    let length = structure.functions.len();
    let lanes = (0..length)
        .find(|&index| {
            matches!(
                structure.functions.get(index),
                Some(Function::Reshape(reshape)) if reshape.shape.rank() == 5
            )
        })
        .expect("the pool records its lanes reshape");

    let wanted = vec![true; length];
    let mut readable = vec![false; length];
    readable[lanes] = true;
    let view = full_view(&network, &wanted, &readable);
    let catalog = Catalog::elect(&Candidates::discover(&view), |_| true);
    assert_eq!(catalog.groups(), 0);
}

#[test]
fn the_home_repertoire_never_elects_the_raise_only_pattern() {
    // Discovery pools the pool candidate on every plan; the home
    // consumer's repertoire excludes it (raise-only), so home runs
    // execute the recorded fold while a total election raises it.
    let (network, root) = pool_network();
    let length = network.structure().functions.len();
    let wanted = vec![true; length];
    let readable = vec![false; length];
    let view = full_view(&network, &wanted, &readable);
    let candidates = Candidates::discover(&view);

    let total = Catalog::elect(&candidates, |_| true);
    assert!(total.at(root).is_some());
    assert!((0..length).any(|index| total.interior(index)));

    let home = Catalog::elect(&candidates, |pattern| pattern.fused().is_some());
    assert_eq!(home.groups(), 0);
    assert!((0..length).all(|index| !home.interior(index)));
}
