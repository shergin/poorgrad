use crate::{Backend, Coverage, Dispatch, Fidelity, Formula, Numerics, Precision};

#[test]
fn the_matrix_pins_every_hardware_row() {
    // Accelerate is the one hardware implementer with kernels for
    // both leaf formulas and the batch-norm task, at both
    // precisions.
    for precision in Precision::ALL {
        assert!(Backend::Accelerate.serves(Formula::Gemm, *precision));
        assert!(Backend::Accelerate.serves(Formula::Map, *precision));
        assert!(Backend::Accelerate.serves(Formula::BatchNormTraining, *precision));
        assert!(
            !Backend::Accelerate
                .coverage(Formula::BatchNormTraining)
                .meets(Fidelity::BitIdentical)
        );
    }
    // Metal has no `f64` at all.
    assert!(Backend::Metal.serves(Formula::Gemm, Precision::F32));
    assert!(Backend::Metal.serves(Formula::Map, Precision::F32));
    assert!(!Backend::Metal.serves(Formula::Gemm, Precision::F64));
    assert!(!Backend::Metal.serves(Formula::Map, Precision::F64));
    // The gemm-only rungs.
    for backend in [Backend::Cuda, Backend::Simd] {
        for precision in Precision::ALL {
            assert!(backend.serves(Formula::Gemm, *precision));
            assert!(!backend.serves(Formula::Map, *precision));
        }
    }
    // No other hardware implementer serves a composed formula, and
    // nobody but the translation column serves the raise-only ones.
    for backend in [
        Backend::Accelerate,
        Backend::Metal,
        Backend::Cuda,
        Backend::Simd,
    ] {
        for formula in [
            Formula::WindowProduct,
            Formula::ReduceWindow,
            Formula::BatchNormInference,
        ] {
            assert_eq!(backend.coverage(formula), Coverage::Absent);
        }
    }
    for backend in [Backend::Metal, Backend::Cuda, Backend::Simd] {
        assert_eq!(
            backend.coverage(Formula::BatchNormTraining),
            Coverage::Absent
        );
    }
}

#[test]
fn fused_serves_its_kernels_bit_identically() {
    // Every in-process kernel either reduces in the recorded order
    // or falls back to composing the recorded formula, so every cell
    // clears bit identity; the envelope enters only through an
    // admitted hardware kernel.
    for formula in [
        Formula::WindowProduct,
        Formula::ReduceWindow,
        Formula::BatchNormTraining,
    ] {
        assert_eq!(
            Backend::Fused.coverage(formula),
            Coverage::Serves {
                fidelity: Fidelity::BitIdentical,
                precisions: Precision::ALL
            }
        );
    }
    for formula in [Formula::Gemm, Formula::Map, Formula::BatchNormInference] {
        assert_eq!(Backend::Fused.coverage(formula), Coverage::Absent);
    }
}

#[test]
fn the_stablehlo_column_is_total() {
    // Emission's repertoire reads this column; totality here is what
    // keeps every pattern raising on every posture.
    for formula in Formula::ALL {
        let cell = Backend::StableHlo.coverage(*formula);
        assert!(cell.serves(), "{formula:?} does not lower");
        assert!(!cell.meets(Fidelity::BitIdentical));
    }
}

#[test]
fn the_fidelity_rule_is_one_comparison() {
    assert!(Fidelity::BitIdentical.meets(Fidelity::BitIdentical));
    assert!(Fidelity::BitIdentical.meets(Fidelity::Envelope));
    assert!(Fidelity::Envelope.meets(Fidelity::Envelope));
    assert!(!Fidelity::Envelope.meets(Fidelity::BitIdentical));
    // The postures demand exactly the two fidelities.
    assert_eq!(Numerics::Exact.fidelity(), Fidelity::BitIdentical);
    assert_eq!(Numerics::Fast.fidelity(), Fidelity::Envelope);
}

#[test]
fn dispatch_names_each_execution_context() {
    for backend in [
        Backend::Accelerate,
        Backend::Metal,
        Backend::Cuda,
        Backend::Simd,
    ] {
        assert_eq!(backend.dispatch(), Dispatch::Offered);
    }
    assert_eq!(Backend::Fused.dispatch(), Dispatch::Elected);
    assert_eq!(Backend::StableHlo.dispatch(), Dispatch::Translated);
}
