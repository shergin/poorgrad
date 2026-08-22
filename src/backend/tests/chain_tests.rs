use crate::{BatchNormTask, GemmTask, MapOperation};

use super::{Formula, MapTask, Numerics, NumericsScope, Precision, offered};

/// A square extent whose product's `2 * m * n * k` flops reach every
/// backend's gemm threshold (the highest is metal's `1 << 25`,
/// exactly this job's count), so any available chain member accepts.
const GEMM_EXTENT: usize = 256;

/// Enough elements to clear every backend's map gate (the highest is
/// metal's `1 << 19` behind accelerate).
const MAP_LENGTH: usize = 1 << 19;

/// Returns whether any member of the formula's chain is available in
/// this build on this machine.
fn a_member_is_available(formula: Formula, precision: Precision) -> bool {
    formula
        .chain(precision)
        .iter()
        .any(|backend| backend.status().is_ok())
}

#[test]
fn the_chain_accepts_exactly_when_a_member_is_available() {
    // The acceptance gate: the canonical jobs clear every member's
    // cost threshold, so an available member that failed to accept
    // would mean a chain entry with no kernel arm behind it — the
    // one drift the declared data cannot exclude by construction.
    // With no member available (the default build among others), the
    // chain must answer `None`: the seam's fixed point.
    let a32 = vec![0.5_f32; GEMM_EXTENT * GEMM_EXTENT];
    let b32 = a32.clone();
    let task32 = GemmTask::new(
        &a32,
        [GEMM_EXTENT, 1],
        &b32,
        [GEMM_EXTENT, 1],
        GEMM_EXTENT,
        GEMM_EXTENT,
        GEMM_EXTENT,
    );
    assert_eq!(
        offered(&task32).is_some(),
        a_member_is_available(Formula::Gemm, Precision::F32)
    );

    let a64 = vec![0.5_f64; GEMM_EXTENT * GEMM_EXTENT];
    let b64 = a64.clone();
    let task64 = GemmTask::new(
        &a64,
        [GEMM_EXTENT, 1],
        &b64,
        [GEMM_EXTENT, 1],
        GEMM_EXTENT,
        GEMM_EXTENT,
        GEMM_EXTENT,
    );
    assert_eq!(
        offered(&task64).is_some(),
        a_member_is_available(Formula::Gemm, Precision::F64)
    );

    let elements32 = vec![0.5_f32; MAP_LENGTH];
    assert_eq!(
        offered(&MapTask::new(MapOperation::Tanh, &elements32)).is_some(),
        a_member_is_available(Formula::Map, Precision::F32)
    );

    let elements64 = vec![0.5_f64; MAP_LENGTH];
    assert_eq!(
        offered(&MapTask::new(MapOperation::Tanh, &elements64)).is_some(),
        a_member_is_available(Formula::Map, Precision::F64)
    );

    // The batch-norm canonical: 64 x 64 sits exactly at the vDSP
    // threshold, so any available member accepts.
    let normalize32: Vec<f32> = (0..64 * 64)
        .map(|index| ((index * 7 % 23) as f32 - 11.0) / 4.0)
        .collect();
    let ones32 = vec![1.0_f32; 64];
    assert_eq!(
        offered(&BatchNormTask::new(
            &normalize32,
            &ones32,
            &ones32,
            1.0e-5_f32,
            64,
            64
        ))
        .is_some(),
        a_member_is_available(Formula::BatchNormTraining, Precision::F32)
    );
    let normalize64: Vec<f64> = normalize32.iter().map(|&value| value as f64).collect();
    let ones64 = vec![1.0_f64; 64];
    assert_eq!(
        offered(&BatchNormTask::new(
            &normalize64,
            &ones64,
            &ones64,
            1.0e-5_f64,
            64,
            64
        ))
        .is_some(),
        a_member_is_available(Formula::BatchNormTraining, Precision::F64)
    );
}

#[test]
fn the_exact_posture_admits_no_envelope_kernel() {
    // Every chain member's cell is envelope-fidelity today, so under
    // `Exact` the fidelity rule empties the chain and the reference paths
    // compute — the old decline-by-fiat, now a consequence.
    let _scope = NumericsScope::enter(Numerics::Exact);
    let a = vec![0.5_f32; GEMM_EXTENT * GEMM_EXTENT];
    let b = a.clone();
    let task = GemmTask::new(
        &a,
        [GEMM_EXTENT, 1],
        &b,
        [GEMM_EXTENT, 1],
        GEMM_EXTENT,
        GEMM_EXTENT,
        GEMM_EXTENT,
    );
    assert_eq!(offered(&task), None);
    let elements = vec![0.5_f32; MAP_LENGTH];
    assert_eq!(offered(&MapTask::new(MapOperation::Tanh, &elements)), None);
}
