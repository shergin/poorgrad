use crate::Shape;

use super::{kaiming, normal, uniform, xavier};

/// Returns the mean and standard deviation of `values`.
fn moments(values: &[f64]) -> (f64, f64) {
    let count = values.len() as f64;
    let mean = values.iter().sum::<f64>() / count;
    let variance = values
        .iter()
        .map(|value| (value - mean) * (value - mean))
        .sum::<f64>()
        / count;
    (mean, variance.sqrt())
}

#[test]
fn same_seed_reproduces_and_seeds_differ() {
    let shape = Shape::new([4, 4]);
    assert_eq!(uniform(7, 1.0)(&shape), uniform(7, 1.0)(&shape));
    assert_ne!(uniform(7, 1.0)(&shape), uniform(8, 1.0)(&shape));
}

#[test]
fn uniform_fills_within_the_scale() {
    let tensor = uniform(7, 0.25)(&Shape::new([1000]));
    let values = tensor.to_vec();
    assert!(values.iter().all(|value| value.abs() <= 0.25));
    // The fill spreads across the range rather than collapsing.
    assert!(values.iter().any(|&value| value > 0.2));
    assert!(values.iter().any(|&value| value < -0.2));
}

#[test]
fn normal_matches_its_moments() {
    let tensor = normal(7, 2.0)(&Shape::new([10000]));
    let (mean, deviation) = moments(&tensor.to_vec());
    assert!(mean.abs() < 0.1);
    assert!((deviation - 2.0).abs() < 0.1);
}

#[test]
fn xavier_bounds_weights_by_both_fans_and_zeroes_biases() {
    let mut initializer = xavier(7);
    // For 300 inputs and 300 outputs the bound is `sqrt(6 / 600) = 0.1`.
    let weights = initializer(&Shape::new([300, 300]));
    assert!(weights.to_vec().iter().all(|value| value.abs() <= 0.1));
    assert!(weights.to_vec().iter().any(|value| value.abs() > 0.05));

    let bias = initializer(&Shape::new([300]));
    assert!(bias.to_vec().iter().all(|&value| value == 0.0));
}

#[test]
fn kaiming_scales_weights_by_fan_in_and_zeroes_biases() {
    let mut initializer = kaiming(7);
    // For 200 inputs the deviation is `sqrt(2 / 200) = 0.1`.
    let weights = initializer(&Shape::new([200, 50]));
    let (mean, deviation) = moments(&weights.to_vec());
    assert!(mean.abs() < 0.01);
    assert!((deviation - 0.1).abs() < 0.01);

    let bias = initializer(&Shape::new([50]));
    assert!(bias.to_vec().iter().all(|&value| value == 0.0));
}

#[test]
#[should_panic(expected = "expects rank-2 weights or rank-1 biases")]
fn fan_aware_initializers_reject_other_ranks() {
    xavier(7)(&Shape::new([2, 3, 4]));
}
