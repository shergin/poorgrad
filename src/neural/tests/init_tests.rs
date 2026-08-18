use crate::{Shape, Tensor};

use super::super::Activation;
use super::{kaiming, normal, scaled, uniform, xavier};

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
    assert_eq!(
        uniform::<f64>(7, 1.0)(&shape),
        uniform::<f64>(7, 1.0)(&shape)
    );
    assert_ne!(
        uniform::<f64>(7, 1.0)(&shape),
        uniform::<f64>(8, 1.0)(&shape)
    );
}

#[test]
fn uniform_fills_within_the_scale() {
    let tensor = uniform::<f64>(7, 0.25)(&Shape::new([1000]));
    let values = tensor.to_vec();
    assert!(values.iter().all(|value| value.abs() <= 0.25));
    // The fill spreads across the range rather than collapsing.
    assert!(values.iter().any(|&value| value > 0.2));
    assert!(values.iter().any(|&value| value < -0.2));
}

#[test]
fn normal_matches_its_moments() {
    let tensor = normal::<f64>(7, 2.0)(&Shape::new([10000]));
    let (mean, deviation) = moments(&tensor.to_vec());
    assert!(mean.abs() < 0.1);
    assert!((deviation - 2.0).abs() < 0.1);
}

#[test]
fn xavier_bounds_weights_by_both_fans_and_zeroes_biases() {
    let mut initializer = xavier::<f64>(7);
    // For 300 inputs and 300 outputs the bound is `sqrt(6 / 600) = 0.1`.
    let weights = initializer(&Shape::new([300, 300]));
    assert!(weights.to_vec().iter().all(|value| value.abs() <= 0.1));
    assert!(weights.to_vec().iter().any(|value| value.abs() > 0.05));

    let bias = initializer(&Shape::new([300]));
    assert!(bias.to_vec().iter().all(|&value| value == 0.0));
}

#[test]
fn kaiming_scales_weights_by_fan_in_and_zeroes_biases() {
    let mut initializer = kaiming::<f64>(7);
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
    xavier::<f64>(7)(&Shape::new([2, 3, 4]));
}

#[test]
fn seeded_streams_are_pinned_forever() {
    // Bits captured from the concrete-f64 implementation before the
    // factories went generic (2026-08-02). The `f64` path must stay
    // bit-identical to them forever, and the `f32` path is the same
    // stream rounded once per element.
    let goldens: [(&[u64; 4], Tensor<f64>, Tensor<f32>); 4] = [
        (
            &[
                0xbfcc341e1ba6cdf8,
                0xbfeeecf0ca02f0e8,
                0x3fe9a610202eac4a,
                0x3fc53aeb70673e28,
            ],
            uniform::<f64>(7, 1.0)(&Shape::new([4])),
            uniform::<f32>(7, 1.0)(&Shape::new([4])),
        ),
        (
            &[
                0x3fffa194ec47d228,
                0xc00dd3fde5949e97,
                0x3f800ea29c8645d7,
                0xbff0efc91ba890b4,
            ],
            normal::<f64>(7, 2.0)(&Shape::new([4])),
            normal::<f32>(7, 2.0)(&Shape::new([4])),
        ),
        (
            &[
                0xbfd14566a85c9c0a,
                0xbff2f01dbe7ab6a2,
                0x3fef69c081567c5c,
                0x3fca0063d7a5b7aa,
            ],
            xavier::<f64>(7)(&Shape::new([2, 2])),
            xavier::<f32>(7)(&Shape::new([2, 2])),
        ),
        (
            &[
                0x3fefa194ec47d228,
                0xbffdd3fde5949e97,
                0x3f700ea29c8645d7,
                0xbfe0efc91ba890b4,
            ],
            kaiming::<f64>(7)(&Shape::new([2, 2])),
            kaiming::<f32>(7)(&Shape::new([2, 2])),
        ),
    ];
    for (golden, doubles, singles) in goldens {
        for ((bits, double), single) in golden.iter().zip(doubles.to_vec()).zip(singles.to_vec()) {
            assert_eq!(double.to_bits(), *bits);
            assert_eq!(single.to_bits(), (f64::from_bits(*bits) as f32).to_bits());
        }
    }
}

#[test]
fn scaled_matches_the_gain_over_the_fan() {
    // For 400 inputs and tanh's gain the deviation is `(5/3) / 20`.
    let mut initializer = scaled::<f64>(7, Activation::Tanh.gain());
    let weights = initializer(&Shape::new([400, 25]));
    let (mean, deviation) = moments(&weights.to_vec());
    assert!(mean.abs() < 0.01);
    assert!((deviation - (5.0 / 3.0) / 20.0).abs() < 0.005);

    let bias = initializer(&Shape::new([25]));
    assert!(bias.to_vec().iter().all(|&value| value == 0.0));
}

#[test]
fn scaled_at_relu_gain_matches_kaiming_statistically() {
    // The named classic is `scaled` at `sqrt(2)`; the formulas round
    // differently in the last bit, so the agreement is statistical,
    // never bitwise — kaiming's seeded outputs stay frozen.
    let weights = scaled::<f64>(7, Activation::Relu.gain())(&Shape::new([200, 50]));
    let classic = kaiming::<f64>(7)(&Shape::new([200, 50]));
    let (_, scaled_deviation) = moments(&weights.to_vec());
    let (_, classic_deviation) = moments(&classic.to_vec());
    assert!((scaled_deviation - classic_deviation).abs() < 0.005);
}

#[test]
#[should_panic(expected = "rank-2 weights or rank-1 biases")]
fn scaled_rejects_higher_ranks() {
    scaled::<f64>(7, 1.0)(&Shape::new([2, 2, 2]));
}

#[test]
fn gains_state_the_standard_factors() {
    assert_eq!(Activation::Tanh.gain(), 5.0 / 3.0);
    assert_eq!(Activation::Relu.gain(), 2.0_f64.sqrt());
}
