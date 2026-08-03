use super::{Backend, BackendUnavailable};

#[test]
fn all_lists_every_backend_in_chain_order() {
    assert_eq!(
        Backend::ALL,
        &[
            Backend::Accelerate,
            Backend::Metal,
            Backend::Cuda,
            Backend::Simd
        ]
    );
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
