use crate::{Tape, Tensor};

use super::Bf16;

#[test]
fn every_bit_pattern_round_trips_through_f32() {
    // Every bf16 value is exactly representable as an `f32`, so the
    // expansion followed by rounding must reproduce the bits; the
    // only exception is NaN, which must stay NaN.
    for bits in 0..=u16::MAX {
        let value = Bf16::from_bits(bits);
        let round_tripped = Bf16::from_f32(value.to_f32());
        if value.to_f32().is_nan() {
            assert!(round_tripped.to_f32().is_nan(), "NaN lost at {bits:#06X}");
            continue;
        }
        assert_eq!(
            round_tripped.to_bits(),
            bits,
            "bit pattern {bits:#06X} did not round-trip"
        );
    }
}

/// The tie literals are written to their full exact decimal: each one
/// names a midpoint the test is about, and shortening it to the digits
/// `f32` needs to round-trip would hide the tie the comments describe.
#[allow(clippy::excessive_precision)]
#[test]
fn rounding_is_nearest_with_ties_to_even() {
    // 1 + 2^-8 sits exactly between 1.0 (even mantissa) and
    // 1 + 2^-7 (odd); the tie must answer 1.0.
    assert_eq!(Bf16::from_f32(1.003_906_25), Bf16::from_f32(1.0));
    // 1 + 3 * 2^-8 sits between 1 + 2^-7 (odd) and 1 + 2^-6 (even);
    // the tie must answer the even neighbor.
    assert_eq!(Bf16::from_f32(1.011_718_75), Bf16::from_f32(1.015_625));
    // Off a tie, rounding goes to the nearest neighbor.
    assert_eq!(Bf16::from_f32(1.006), Bf16::from_f32(1.007_812_5));
}

#[test]
fn rounding_carries_into_the_extremes() {
    // `f32::MAX` lies beyond the last finite bf16 plus half a step,
    // so the rounding carry must propagate into infinity.
    assert_eq!(Bf16::from_f32(f32::MAX).to_f32(), f32::INFINITY);
    assert_eq!(Bf16::from_f32(f32::MIN).to_f32(), f32::NEG_INFINITY);
    // The zero signs survive exactly.
    assert_eq!(Bf16::from_f32(-0.0).to_bits(), 0x8000);
    assert_eq!(Bf16::from_f32(0.0).to_bits(), 0x0000);
    assert!(Bf16::from_f32(f32::NAN).to_f32().is_nan());
}

#[test]
fn counted_is_exact_up_to_256() {
    use crate::{Differentiable, Shape};

    for count in [0, 1, 100, 255, 256] {
        assert_eq!(
            Bf16::counted(Shape::scalar(), count).to_f32(),
            count as f32,
            "count {count} must convert exactly"
        );
    }
    // Above 256 the significand steps by two: 257 ties down to the
    // even 256, 259 ties up to the even 260.
    assert_eq!(Bf16::counted(Shape::scalar(), 257).to_f32(), 256.0);
    assert_eq!(Bf16::counted(Shape::scalar(), 259).to_f32(), 260.0);
}

#[test]
fn each_operation_rounds_the_f32_result_once() {
    let third = Bf16::from_f32(1.0) / Bf16::from_f32(3.0);
    assert_eq!(third, Bf16::from_f32(1.0_f32 / 3.0));
    // The classic swamping case: past 256 an added one no longer
    // lands, which is exactly the per-op contract.
    let large = Bf16::from_f32(256.0);
    assert_eq!(large + Bf16::ONE, large);
}

#[test]
fn everyday_numeric_traits_behave_like_floats() {
    // Display prints the exact expansion, like any float.
    assert_eq!(format!("{}", Bf16::from_f32(1.5)), "1.5");
    assert_eq!(format!("{}", Bf16::from_f32(-2.0)), "-2");
    // Ordering has float semantics: a NaN orders against nothing.
    assert!(Bf16::from_f32(1.0) < Bf16::from_f32(2.0));
    assert!(Bf16::from_f32(f32::NAN).partial_cmp(&Bf16::ONE).is_none());
    // The default is the additive identity.
    assert_eq!(Bf16::default(), Bf16::ZERO);
    // An `f64` rounds once (double rounding through `f32` is exact
    // at bf16's precision), and widening back is exact.
    assert_eq!(Bf16::from(1.0_f64 / 3.0), Bf16::from_f32(1.0_f32 / 3.0));
    assert_eq!(f64::from(Bf16::from_f32(1.5)), 1.5_f64);
}

#[test]
fn negation_flips_only_the_sign_bit() {
    assert_eq!(-Bf16::from_f32(1.5), Bf16::from_f32(-1.5));
    assert_eq!((-Bf16::ZERO).to_bits(), 0x8000);
    assert!((-Bf16::from_f32(f32::NAN)).to_f32().is_nan());
}

#[test]
fn maximum_and_step_answer_exactly() {
    use crate::Elementary;

    let smaller = Bf16::from_f32(1.5);
    let larger = Bf16::from_f32(2.5);
    assert_eq!(smaller.maximum(&larger), larger);
    assert_eq!(larger.step(&smaller), Bf16::ONE);
    assert_eq!(smaller.step(&larger), Bf16::ZERO);
    // Ties answer one, matching the documented `step` contract.
    assert_eq!(smaller.step(&smaller), Bf16::ONE);
}

#[test]
fn matmul_accumulates_in_f32_and_rounds_once() {
    use crate::Tensorial;

    // Per-op bf16 accumulation would answer 256: the running total
    // swamps each added one. The pinned contract accumulates in f32
    // (257, then 258) and rounds once, and 258 is exactly a bf16.
    let left = Tensor::new([1, 3], [256.0_f32, 1.0, 1.0].map(Bf16::from_f32).to_vec());
    let right = Tensor::new([3, 1], [1.0_f32, 1.0, 1.0].map(Bf16::from_f32).to_vec());
    let product = left.matmul(&right);
    assert_eq!(product.to_vec(), vec![Bf16::from_f32(258.0)]);
}

#[test]
fn matmul_accumulation_is_representation_independent() {
    use crate::{Tensor, Tensorial};

    // A constant operand has no gemm operand and takes the composed
    // path; the accumulation contract must answer identically there,
    // or bf16 results would depend on the storage representation.
    let ones_constant = Tensor::filled([1, 3], Bf16::ONE);
    let ones_dense = Tensor::new([1, 3], [1.0_f32; 3].map(Bf16::from_f32).to_vec());
    let right = Tensor::new([3, 1], [256.0_f32, 1.0, 1.0].map(Bf16::from_f32).to_vec());
    assert_eq!(
        ones_constant.matmul(&right).to_vec(),
        ones_dense.matmul(&right).to_vec(),
    );
    assert_eq!(
        ones_constant.matmul(&right).to_vec(),
        vec![Bf16::from_f32(258.0)]
    );
}

#[test]
fn reductions_accumulate_in_f32() {
    use crate::Tensorial;

    // Per-op bf16 summation would swamp at 256; the accumulator
    // contract sums in f32 and rounds once.
    let values = Tensor::new([3], [256.0_f32, 1.0, 1.0].map(Bf16::from_f32).to_vec());
    assert_eq!(values.sum().to_vec(), vec![Bf16::from_f32(258.0)]);
    let rows = Tensor::new([1, 3], [256.0_f32, 1.0, 1.0].map(Bf16::from_f32).to_vec());
    assert_eq!(rows.sum_along(1).to_vec(), vec![Bf16::from_f32(258.0)]);
}

#[test]
fn scatter_accumulates_duplicate_rows_in_f32() {
    use crate::Tensorial;

    // Three gradient rows land on the same vocabulary row; their sum
    // accumulates in f32, so the ones keep landing past 256.
    let gradient = Tensor::new([3, 1], [256.0_f32, 1.0, 1.0].map(Bf16::from_f32).to_vec());
    let selection = Tensor::selection(vec![0, 0, 0], 1, Bf16::ONE);
    let folded = gradient.scatter(&selection, 1);
    assert_eq!(folded.to_vec(), vec![Bf16::from_f32(258.0)]);
}

#[test]
fn scalar_networks_differentiate_bf16() {
    let tape = Tape::new();
    let x = tape.parameter(Bf16::from_f32(1.5));
    let loss = (x * x).symbol();
    let x = x.symbol();
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(*run.of(loss), Bf16::from_f32(2.25));

    let gradients = run.backward(loss);
    assert_eq!(*gradients.of(x), Bf16::from_f32(3.0));
}

#[test]
fn convert_crosses_the_precision_boundary_exactly() {
    // Narrowing rounds each element to the nearest bf16; widening a
    // bf16 tensor back is exact, so the round trip is idempotent.
    let singles = Tensor::new([2, 2], vec![1.0_f32, 0.3, -2.5, 300.0]);
    let narrowed: Tensor<Bf16> = singles.convert();
    let expected: Vec<Bf16> = singles.iter().map(Bf16::from_f32).collect();
    assert_eq!(narrowed.to_vec(), expected);
    let widened: Tensor<f32> = narrowed.convert();
    let round_tripped: Tensor<Bf16> = widened.convert();
    assert_eq!(round_tripped.to_vec(), narrowed.to_vec());

    // A constant converts in O(1) and stays a constant: the widened
    // tensor still has no contiguous buffer to lend.
    let constant = Tensor::filled([1024], Bf16::ONE);
    let widened: Tensor<f32> = constant.convert();
    assert!(widened.as_slice().is_none());
    assert_eq!(widened.to_vec(), vec![1.0_f32; 1024]);
}

#[test]
fn bf16_gradients_track_f32_within_epsilon() {
    use crate::Value;

    // The same tiny model recorded in both precisions through
    // `convert`: the bf16 gradients must match the f32 oracle to
    // bf16 precision — the end-to-end mixed-precision contract.
    let weights = Tensor::new([2, 2], vec![0.8_f32, -1.3, 0.4, 2.1]);
    let x = Tensor::new([2, 2], vec![0.5_f32, -1.1, 1.9, 0.7]);

    let oracle_tape = Tape::new();
    let oracle_weights = oracle_tape.parameter(weights.clone());
    let oracle_x = oracle_tape.leaf(x.clone());
    let oracle_loss = oracle_x.matmul(oracle_weights).relu().sum().symbol();
    let oracle_weights = oracle_weights.symbol();
    let oracle = oracle_tape.into_network();
    let oracle_gradients = oracle
        .forward(&oracle.parameters(), [])
        .backward(oracle_loss);
    let expected = oracle_gradients.of(oracle_weights).to_vec();

    let tape: Tape<Tensor<Bf16>> = Tape::new();
    let narrowed_weights: Value<'_, Tensor<Bf16>> = tape.parameter(weights.convert());
    let narrowed_x = tape.leaf(x.convert());
    let loss = narrowed_x.matmul(narrowed_weights).relu().sum().symbol();
    let narrowed_weights = narrowed_weights.symbol();
    let network = tape.into_network();
    let gradients = network.forward(&network.parameters(), []).backward(loss);

    let epsilon = 7.8125e-3_f32;
    for (narrow, wide) in gradients.of(narrowed_weights).iter().zip(expected) {
        assert!(
            (narrow.to_f32() - wide).abs() <= epsilon * (1.0 + wide.abs()),
            "bf16 gradient {narrow:?} strays from the f32 oracle {wide}"
        );
    }
}

#[test]
fn tensor_networks_differentiate_bf16() {
    let tape = Tape::new();
    let elements: Vec<Bf16> = [-2.0, 0.0, 3.0].map(Bf16::from_f32).to_vec();
    let x = tape.leaf(Tensor::new([3], elements));
    let loss = x.abs().sum().symbol();
    let x = x.symbol();
    let network = tape.into_network();

    let run = network.forward(&network.parameters(), []);
    assert_eq!(run.of(loss).to_vec(), vec![Bf16::from_f32(5.0)]);

    let gradients = run.backward(loss);
    let expected: Vec<Bf16> = [-1.0, 1.0, 1.0].map(Bf16::from_f32).to_vec();
    assert_eq!(gradients.of(x).to_vec(), expected);
}
