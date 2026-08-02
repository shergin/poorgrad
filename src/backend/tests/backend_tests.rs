use super::{Backend, BackendUnavailable};

#[test]
fn all_lists_every_backend_in_chain_order() {
    assert_eq!(Backend::ALL, &[Backend::Accelerate]);
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
