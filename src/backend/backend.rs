use std::error::Error;
use std::fmt::{self, Display, Formatter};

use static_assertions::assert_impl_all;

// Compile-time thread-safety contract; the anchor rationale is
// documented in `network.rs`.
assert_impl_all!(Backend: Send, Sync);
assert_impl_all!(BackendUnavailable: Send, Sync);

/// The acceleration backends this crate can be built with.
///
/// The backend chain tries backends in declaration order, so the
/// order here is the documented priority. Variants
/// name concrete implementations — what a build links — so a future
/// backend arrives as a new variant, never as a broadening of an
/// existing one. The enum exists in every build: whether a variant
/// is compiled in is a [`status`](Backend::status) answer, not a
/// compile error, so interrogating the chain never needs a `cfg`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Apple's Accelerate framework: `cblas_sgemm`/`cblas_dgemm`,
    /// executing on the AMX/SME matrix units on Apple Silicon and
    /// AVX kernels on Intel Macs. Behind the `accelerate` feature,
    /// macOS only. Leads the chain: it measured ahead of the Metal
    /// kernel at every size.
    Accelerate,
    /// Hand-written simdgroup-matrix GPU kernels for large `f32`
    /// products (Metal has no `f64`), compiled from source at first
    /// use. Behind the `metal` feature, macOS only; serves what
    /// BLAS declines, and everything large in metal-only builds.
    Metal,
    /// The `matrixmultiply` crate's tuned CPU microkernels with
    /// runtime instruction-set dispatch (AVX-512F, AVX2+FMA, AVX,
    /// NEON), single-threaded. Behind the `simd` feature, every
    /// platform — the portable rung for Linux and everyone else,
    /// and mop-up behind the Apple backends on macOS.
    Simd,
}

impl Backend {
    /// Every backend this crate version defines, in chain order.
    pub const ALL: &'static [Backend] = &[Backend::Accelerate, Backend::Metal, Backend::Simd];

    /// Reports whether this backend would accept work in this build
    /// on this machine, forcing its lazy setup if it has one.
    ///
    /// `NotCompiled` is an ordinary answer, which is what lets a
    /// build without the feature ask the question; a loud program
    /// asserts readiness at startup with
    /// `Backend::Accelerate.status().expect(..)`.
    pub fn status(self) -> Result<(), BackendUnavailable> {
        match self {
            Backend::Metal => {
                if !cfg!(feature = "metal") {
                    return Err(BackendUnavailable::NotCompiled);
                }
                if !cfg!(target_os = "macos") {
                    return Err(BackendUnavailable::PlatformUnsupported);
                }
                #[cfg(all(feature = "metal", target_os = "macos"))]
                {
                    super::metal::status()
                }
                #[cfg(not(all(feature = "metal", target_os = "macos")))]
                {
                    unreachable!("the cfg! guards above cover this build")
                }
            }
            Backend::Accelerate => {
                if !cfg!(feature = "accelerate") {
                    return Err(BackendUnavailable::NotCompiled);
                }
                if !cfg!(target_os = "macos") {
                    return Err(BackendUnavailable::PlatformUnsupported);
                }
                // Accelerate is a link-time dependency with nothing
                // to initialize and nothing to lose at run time.
                Ok(())
            }
            Backend::Simd => {
                if !cfg!(feature = "simd") {
                    return Err(BackendUnavailable::NotCompiled);
                }
                // Pure CPU code with runtime instruction-set
                // dispatch: no platform arm, no device, nothing to
                // initialize and nothing to lose at run time.
                Ok(())
            }
        }
    }
}

/// Why a [`Backend`] would decline all work in this build.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BackendUnavailable {
    /// The backend's cargo feature is off in this build.
    NotCompiled,
    /// The feature is on, but this platform has no such backend.
    PlatformUnsupported,
    /// One-time setup failed; the reason is the message.
    Initialization(String),
    /// Disabled after a runtime error; the reason is the message.
    Poisoned(String),
}

impl Display for BackendUnavailable {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            BackendUnavailable::NotCompiled => {
                write!(
                    formatter,
                    "the backend's cargo feature is off in this build"
                )
            }
            BackendUnavailable::PlatformUnsupported => {
                write!(formatter, "this platform has no such backend")
            }
            BackendUnavailable::Initialization(reason) => {
                write!(formatter, "backend setup failed: {reason}")
            }
            BackendUnavailable::Poisoned(reason) => {
                write!(
                    formatter,
                    "backend disabled after a runtime error: {reason}"
                )
            }
        }
    }
}

impl Error for BackendUnavailable {}

#[cfg(test)]
#[path = "tests/backend_tests.rs"]
mod tests;
