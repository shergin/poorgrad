use crate::{Backend, Bar, Coverage, Dispatch, Formula, Numerics, Precision, Precisions};

#[test]
fn the_matrix_pins_every_hardware_row() {
    // Accelerate is the one hardware implementer with a kernel for
    // both leaf formulas at both precisions.
    for precision in Precision::ALL {
        assert!(Backend::Accelerate.serves(Formula::Gemm, *precision));
        assert!(Backend::Accelerate.serves(Formula::Map, *precision));
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
    // No hardware implementer serves a composed formula.
    for backend in [
        Backend::Accelerate,
        Backend::Metal,
        Backend::Cuda,
        Backend::Simd,
    ] {
        for formula in [
            Formula::WindowProduct,
            Formula::ReduceWindow,
            Formula::BatchNormTraining,
            Formula::BatchNormInference,
        ] {
            assert_eq!(backend.coverage(formula), Coverage::Absent);
        }
    }
}

#[test]
fn fused_serves_the_window_product_bit_identically() {
    assert_eq!(
        Backend::Fused.coverage(Formula::WindowProduct),
        Coverage::Serves {
            bar: Bar::BitIdentical,
            precisions: Precisions::Any
        }
    );
    for formula in [
        Formula::Gemm,
        Formula::Map,
        Formula::ReduceWindow,
        Formula::BatchNormTraining,
        Formula::BatchNormInference,
    ] {
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
        assert!(!cell.clears(Bar::BitIdentical));
    }
}

#[test]
fn the_bar_rule_is_one_comparison() {
    assert!(Bar::BitIdentical.meets(Bar::BitIdentical));
    assert!(Bar::BitIdentical.meets(Bar::Envelope));
    assert!(Bar::Envelope.meets(Bar::Envelope));
    assert!(!Bar::Envelope.meets(Bar::BitIdentical));
    // The postures demand exactly the two bars.
    assert_eq!(Numerics::Exact.bar(), Bar::BitIdentical);
    assert_eq!(Numerics::Fast.bar(), Bar::Envelope);
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

#[test]
fn precisions_admit_their_lists() {
    assert!(Precisions::Any.admit(Precision::F32));
    assert!(Precisions::Any.admit(Precision::F64));
    let only = Precisions::Only(&[Precision::F32]);
    assert!(only.admit(Precision::F32));
    assert!(!only.admit(Precision::F64));
}
