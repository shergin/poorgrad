use crate::function::Function;
use crate::graph::Network;
use crate::{BatchNorm, Tape, Tensor};

use super::super::candidates::Candidates;
use super::super::catalog::Catalog;
use super::super::pattern::Pattern;
use super::super::view::View;

/// Records the training-mode expression of a two-feature layer over a
/// three-sample batch and returns the network with the root, mean,
/// and variance indices.
fn training_network() -> (Network<Tensor<f64>>, usize, usize, usize) {
    let tape = Tape::new();
    let layer = BatchNorm::new(
        &tape,
        Tensor::new([2], [1.0_f64, 0.5]),
        Tensor::new([2], [0.0, -0.25]),
        Tensor::filled([], 1e-5),
    );
    let input = tape.leaf(Tensor::new(
        [3, 2],
        (0..6).map(|v| v as f64 * 0.7 - 2.0).collect::<Vec<_>>(),
    ));
    let normalization = layer.express(&tape, input);
    let root = normalization.output.symbol().id.index();
    let mean = normalization.mean.symbol().id.index();
    let variance = normalization.variance.symbol().id.index();
    (tape.into_network(), root, mean, variance)
}

/// Builds a view over the whole network with every node wanted.
fn full_view<'plan>(
    network: &'plan Network<Tensor<f64>>,
    wanted: &'plan [bool],
    readable: &'plan [bool],
) -> View<'plan, Tensor<f64>> {
    View::new(network.structure(), wanted, readable)
}

#[test]
fn the_training_formula_matches_with_observed_statistics() {
    // Observing the mean and variance does not bar the match: they
    // are named results of the raise, allowed in the keep-set.
    let (network, root, mean, variance) = training_network();
    let length = network.structure().functions.len();
    let wanted = vec![true; length];
    let mut readable = vec![false; length];
    readable[root] = true;
    readable[mean] = true;
    readable[variance] = true;
    let view = full_view(&network, &wanted, &readable);
    let catalog = Catalog::elect(&Candidates::discover(&view), |_| true);

    let Some(Pattern::BatchNormTraining(group)) = catalog.at(root) else {
        panic!("the training formula matches at the shift root");
    };
    assert_eq!(group.mean, mean);
    assert_eq!(group.variance, variance);
    // The named statistics skip alongside the unnamed interiors in
    // the raising consumer's election.
    assert!(catalog.interior(mean));
    assert!(catalog.interior(variance));
}

#[test]
fn an_observed_centering_bars_the_training_match() {
    // The centering is an unnamed interior: keeping it readable
    // requires the primitive path, for both variants — its fan-out
    // sits inside the diamond, so no closed candidate can hold it.
    let (network, root, _mean, _variance) = training_network();
    let structure = network.structure();
    let length = structure.functions.len();
    let centered = (0..length)
        .find(|&index| matches!(structure.functions.get(index), Some(Function::Sub(_))))
        .expect("the formula records its centering");

    let wanted = vec![true; length];
    let mut readable = vec![false; length];
    readable[root] = true;
    readable[centered] = true;
    let view = full_view(&network, &wanted, &readable);
    let catalog = Catalog::elect(&Candidates::discover(&view), |_| true);
    assert!(catalog.at(root).is_none());
}

#[test]
fn a_shared_statistic_bars_the_match() {
    // A mean that also feeds an unrelated wanted expression is not
    // the training raise's private result, so the training variant
    // rejects. The inference variant rejects too: in a training
    // recording the centering feeds the variance computation, a
    // consumer outside the inference tail, so its closure fails.
    let tape = Tape::new();
    let layer = BatchNorm::new(
        &tape,
        Tensor::new([2], [1.0_f64, 0.5]),
        Tensor::new([2], [0.0, -0.25]),
        Tensor::filled([], 1e-5),
    );
    let input = tape.leaf(Tensor::new(
        [3, 2],
        (0..6).map(|v| v as f64 * 0.4 - 1.0).collect::<Vec<_>>(),
    ));
    let normalization = layer.express(&tape, input);
    let _drift = normalization.mean.sum();
    let root = normalization.output.symbol().id.index();
    let network = tape.into_network();

    let length = network.structure().functions.len();
    let wanted = vec![true; length];
    let readable = vec![false; length];
    let view = full_view(&network, &wanted, &readable);
    let catalog = Catalog::elect(&Candidates::discover(&view), |_| true);
    assert!(catalog.at(root).is_none());
}

#[test]
fn an_unverified_divisor_is_not_a_training_mean() {
    // The unbiased-variance spelling divides by `batch - 1`: the
    // count leaf fails `is_counted`, so the formula must not raise as
    // a training batch norm — `batch_norm_training` computes the
    // biased statistic. It stays primitive entirely: the centering
    // feeds the variance computation, so the inference tail is not
    // closed either.
    let tape = Tape::new();
    let scale = tape.parameter(Tensor::new([2], [1.0_f64, 1.0]));
    let shift = tape.parameter(Tensor::new([2], [0.0_f64, 0.0]));
    let epsilon = tape.leaf(Tensor::filled([], 1e-5_f64));
    let input = tape.leaf(Tensor::new(
        [3, 2],
        (0..6).map(|v| v as f64 * 0.9 - 2.5).collect::<Vec<_>>(),
    ));
    let mean = input.mean_along(0);
    let centered = input - mean.broadcast_along(0, input);
    let variance = (centered * centered).sum_along(0) / Tensor::filled([2], 2.0);
    let deviation = (variance + epsilon.broadcast_like(variance)).sqrt();
    let normalized = centered / deviation.broadcast_along(0, centered);
    let output =
        normalized * scale.broadcast_along(0, centered) + shift.broadcast_along(0, centered);
    let root = output.symbol().id.index();
    let network = tape.into_network();

    let length = network.structure().functions.len();
    let wanted = vec![true; length];
    let readable = vec![false; length];
    let view = full_view(&network, &wanted, &readable);
    let catalog = Catalog::elect(&Candidates::discover(&view), |_| true);
    assert!(catalog.at(root).is_none());
}

#[test]
fn the_inference_formula_matches_supplied_statistics() {
    let tape = Tape::new();
    let layer = BatchNorm::new(
        &tape,
        Tensor::new([2], [1.0_f64, 0.5]),
        Tensor::new([2], [0.0, -0.25]),
        Tensor::filled([], 1e-5),
    );
    let input = tape.input(Tensor::new(
        [3, 2],
        (0..6).map(|v| v as f64 * 0.7 - 2.0).collect::<Vec<_>>(),
    ));
    let mean = tape.input(Tensor::new([2], [0.1_f64, -0.1]));
    let variance = tape.input(Tensor::new([2], [1.5_f64, 0.5]));
    let output = layer.express_with(&tape, input, mean, variance);
    let root = output.symbol().id.index();
    let mean = mean.symbol().id.index();
    let variance = variance.symbol().id.index();
    let network = tape.into_network();

    let length = network.structure().functions.len();
    let wanted = vec![true; length];
    let readable = vec![false; length];
    let view = full_view(&network, &wanted, &readable);
    let catalog = Catalog::elect(&Candidates::discover(&view), |_| true);
    let Some(Pattern::BatchNormInference(group)) = catalog.at(root) else {
        panic!("the inference formula matches at the shift root");
    };
    // The supplied statistics are extra reads, not named results: they
    // stay ordinary emitted operands.
    assert_eq!(group.mean, mean);
    assert_eq!(group.variance, variance);
    assert!(!catalog.interior(mean));
    assert!(!catalog.interior(variance));
    // Raise-only: the home repertoire never elects a batch norm.
    let home = Catalog::elect(&Candidates::discover(&view), |pattern| {
        pattern.fused().is_some()
    });
    assert_eq!(home.groups(), 0);
}

#[test]
fn closed_rejects_a_readable_node_listed_as_interior() {
    // The named-result refinement is not named-wins: a readable mean
    // listed as an unnamed interior fails the closure outright.
    let (network, root, mean, _variance) = training_network();
    let length = network.structure().functions.len();
    let wanted = vec![true; length];
    let mut readable = vec![false; length];
    readable[mean] = true;
    let view = full_view(&network, &wanted, &readable);
    assert!(!view.closed(root, &[mean], &[]));
}
