use super::{Backend, BackendUnavailable};

#[test]
fn all_lists_every_backend_in_chain_order() {
    assert_eq!(Backend::ALL, &[Backend::Accelerate, Backend::Metal]);
}

#[test]
fn metal_status_reports_the_build() {
    let status = Backend::Metal.status();
    if cfg!(all(feature = "metal", target_os = "macos")) {
        // The lazy setup either succeeds or reports its reason; on
        // real Apple hardware it succeeds.
        assert!(matches!(
            status,
            Ok(()) | Err(BackendUnavailable::Initialization(_))
        ));
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
