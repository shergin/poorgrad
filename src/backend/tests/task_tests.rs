use super::{Backend, TaskKind};

#[test]
fn all_lists_every_kind() {
    assert_eq!(
        TaskKind::ALL,
        &[
            TaskKind::GemmF32,
            TaskKind::GemmF64,
            TaskKind::MapF32,
            TaskKind::MapF64
        ]
    );
}

#[test]
fn chains_declare_the_measured_orders() {
    // The order pins: each chain is a measured decision, so a change
    // here must arrive with a new measurement, not as a side effect.
    assert_eq!(
        TaskKind::GemmF32.chain(),
        &[
            Backend::Accelerate,
            Backend::Metal,
            Backend::Cuda,
            Backend::Simd
        ]
    );
    assert_eq!(
        TaskKind::GemmF64.chain(),
        &[Backend::Accelerate, Backend::Cuda, Backend::Simd]
    );
    assert_eq!(
        TaskKind::MapF32.chain(),
        &[Backend::Metal, Backend::Accelerate]
    );
    assert_eq!(TaskKind::MapF64.chain(), &[Backend::Accelerate]);
}

#[test]
fn chains_hold_known_backends_without_duplicates() {
    for kind in TaskKind::ALL {
        let chain = kind.chain();
        for (index, backend) in chain.iter().enumerate() {
            assert!(Backend::ALL.contains(backend));
            assert!(
                !chain[..index].contains(backend),
                "{backend:?} appears twice in the {kind:?} chain"
            );
        }
    }
}

#[test]
fn serves_answers_designed_coverage() {
    // Accelerate is the one backend with a kernel for every kind.
    for kind in TaskKind::ALL {
        assert!(Backend::Accelerate.serves(*kind));
    }
    // Metal has no `f64` at all.
    assert!(Backend::Metal.serves(TaskKind::GemmF32));
    assert!(Backend::Metal.serves(TaskKind::MapF32));
    assert!(!Backend::Metal.serves(TaskKind::GemmF64));
    assert!(!Backend::Metal.serves(TaskKind::MapF64));
    // The gemm-only rungs: `matrixmultiply` has no map, and a cuda
    // map would be PCIe-bound.
    for backend in [Backend::Cuda, Backend::Simd] {
        assert!(backend.serves(TaskKind::GemmF32));
        assert!(backend.serves(TaskKind::GemmF64));
        assert!(!backend.serves(TaskKind::MapF32));
        assert!(!backend.serves(TaskKind::MapF64));
    }
}
