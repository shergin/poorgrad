use super::{Backend, BackendUnavailable};

#[test]
fn all_lists_every_implementer() {
    assert_eq!(
        Backend::ALL,
        &[
            Backend::Accelerate,
            Backend::Metal,
            Backend::Cuda,
            Backend::Simd,
            Backend::Fused,
            Backend::StableHlo
        ]
    );
}

#[test]
fn compiled_is_the_build_time_half_of_status() {
    // `compiled` answers from build facts alone, so it must agree
    // with `status` exactly on the two build-fact errors and never
    // consult a device.
    for backend in Backend::ALL {
        let build_absent = matches!(
            backend.status(),
            Err(BackendUnavailable::NotCompiled) | Err(BackendUnavailable::PlatformUnsupported)
        );
        assert_eq!(backend.compiled(), !build_absent, "{backend:?}");
    }
}

#[test]
fn the_in_process_implementers_are_always_resident() {
    assert!(Backend::Fused.compiled());
    assert!(Backend::StableHlo.compiled());
    assert_eq!(Backend::Fused.status(), Ok(()));
    assert_eq!(Backend::StableHlo.status(), Ok(()));
}

#[test]
fn cuda_status_reports_the_build() {
    let status = Backend::Cuda.status();
    if cfg!(all(feature = "cuda", target_os = "linux")) {
        // The lazy setup succeeds where the NVIDIA stack exists; the
        // acceptable failures are the two expected environments — no
        // libraries, no device. Every other initialization reason is
        // a broken backend and fails here.
        match status {
            Ok(()) => {}
            Err(BackendUnavailable::Initialization(reason)) => {
                assert!(
                    reason.contains("is not available") || reason == "no CUDA device",
                    "CUDA setup failed: {reason}"
                );
            }
            Err(other) => panic!("unexpected CUDA status: {other}"),
        }
    } else if cfg!(feature = "cuda") {
        assert_eq!(status, Err(BackendUnavailable::PlatformUnsupported));
    } else {
        assert_eq!(status, Err(BackendUnavailable::NotCompiled));
    }
}

#[test]
fn simd_status_reports_the_build() {
    // The simd backend has no platform arm and nothing to
    // initialize: compiled means ready, on every OS.
    let status = Backend::Simd.status();
    if cfg!(feature = "simd") {
        assert_eq!(status, Ok(()));
    } else {
        assert_eq!(status, Err(BackendUnavailable::NotCompiled));
    }
}

#[test]
fn metal_status_reports_the_build() {
    let status = Backend::Metal.status();
    if cfg!(all(feature = "metal", target_os = "macos")) {
        // The lazy setup succeeds on real Apple hardware; the only
        // acceptable failure is a machine without any Metal device
        // (the virtualized CI runners). Every other initialization
        // reason — a shader that does not compile, a missing kernel,
        // a rejected pipeline — is a broken backend and fails here.
        match status {
            Ok(()) => {}
            Err(BackendUnavailable::Initialization(reason)) => {
                assert_eq!(reason, "no Metal device", "Metal setup failed: {reason}");
            }
            Err(other) => panic!("unexpected Metal status: {other}"),
        }
    } else if cfg!(feature = "metal") {
        assert_eq!(status, Err(BackendUnavailable::PlatformUnsupported));
    } else {
        assert_eq!(status, Err(BackendUnavailable::NotCompiled));
    }
}

#[test]
fn status_reports_the_build() {
    let status = Backend::Accelerate.status();
    if cfg!(all(feature = "accelerate", target_os = "macos")) {
        assert_eq!(status, Ok(()));
    } else if cfg!(feature = "accelerate") {
        assert_eq!(status, Err(BackendUnavailable::PlatformUnsupported));
    } else {
        assert_eq!(status, Err(BackendUnavailable::NotCompiled));
    }
}

#[test]
fn unavailability_reasons_display() {
    for reason in [
        BackendUnavailable::NotCompiled,
        BackendUnavailable::PlatformUnsupported,
        BackendUnavailable::Initialization("no device".into()),
        BackendUnavailable::Poisoned("command buffer error".into()),
    ] {
        assert!(!reason.to_string().is_empty());
    }
}
