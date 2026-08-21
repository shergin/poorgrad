use super::super::coverage::Dispatch;
use super::{Backend, Formula, Precision};

#[test]
fn all_lists_every_formula() {
    assert_eq!(
        Formula::ALL,
        &[
            Formula::Gemm,
            Formula::Map,
            Formula::WindowProduct,
            Formula::ReduceWindow,
            Formula::BatchNormTraining,
            Formula::BatchNormInference
        ]
    );
    assert_eq!(Precision::ALL, &[Precision::F32, Precision::F64]);
}

#[test]
fn chains_declare_the_measured_orders() {
    // The order pins: each chain is a measured decision, so a change
    // here must arrive with a new measurement, not as a side effect.
    assert_eq!(
        Formula::Gemm.chain(Precision::F32),
        &[
            Backend::Accelerate,
            Backend::Metal,
            Backend::Cuda,
            Backend::Simd
        ]
    );
    assert_eq!(
        Formula::Gemm.chain(Precision::F64),
        &[Backend::Accelerate, Backend::Cuda, Backend::Simd]
    );
    assert_eq!(
        Formula::Map.chain(Precision::F32),
        &[Backend::Metal, Backend::Accelerate]
    );
    assert_eq!(Formula::Map.chain(Precision::F64), &[Backend::Accelerate]);
}

#[test]
fn composed_formulas_have_no_offer_chain() {
    // Their kernels are elected, never offered buffers — until one
    // earns a job face.
    for formula in [
        Formula::WindowProduct,
        Formula::ReduceWindow,
        Formula::BatchNormTraining,
        Formula::BatchNormInference,
    ] {
        for precision in Precision::ALL {
            assert!(formula.chain(*precision).is_empty());
        }
    }
}

#[test]
fn chains_agree_with_the_coverage_matrix() {
    // Membership is not free-standing data: a backend appears in a
    // formula's chain exactly when it is offer-dispatched and its
    // coverage cell admits the precision. Order is the chains' only
    // own contribution.
    for formula in Formula::ALL {
        for precision in Precision::ALL {
            let chain = formula.chain(*precision);
            for backend in Backend::ALL {
                let in_chain = chain.contains(backend);
                let offered_server =
                    backend.dispatch() == Dispatch::Offered && backend.serves(*formula, *precision);
                assert_eq!(
                    in_chain, offered_server,
                    "{backend:?} vs the {formula:?}/{precision:?} chain"
                );
            }
            for (index, backend) in chain.iter().enumerate() {
                assert!(
                    !chain[..index].contains(backend),
                    "{backend:?} appears twice in the {formula:?}/{precision:?} chain"
                );
            }
        }
    }
}
