use std::error::Error;
use std::fmt::{self, Display, Formatter};

use static_assertions::assert_impl_all;

use super::accelerate::Accelerate;
use super::coverage::{Cell, Dispatch};
use super::cuda::Cuda;
use super::formula::{Formula, Precision};
use super::fused::Fused;
use super::implementer::Implementer;
use super::metal::Metal;
use super::simd::Simd;
use super::stablehlo::StableHlo;

// Request-time thread-safety contract; the anchor rationale is
// documented in `network.rs`.
assert_impl_all!(Backend: Send, Sync);
assert_impl_all!(BackendUnavailable: Send, Sync);

/// The implementers of named formulas: everything that can serve
/// work faster than the reference paths, in LLVM's sense of the
/// word — hardware kernel providers, the crate's own fused kernels,
/// and the translation library alike.
///
/// Variants name concrete implementations, so a future implementer
/// arrives as a new variant, never as a broadening of an existing
/// one. The enum is the public axis; every answer delegates to the
/// variant's descriptor — a unit struct implementing the
/// crate-internal `Implementer` contract in the backend's own
/// module, always compiled, while its kernels sit behind the
/// feature `cfg`. The enum exists in every build: every question is
/// an answer, not a compile error, so interrogating the stack never
/// needs a `cfg`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Apple's Accelerate framework: `cblas_sgemm`/`cblas_dgemm`,
    /// executing on the AMX/SME matrix units on Apple Silicon and
    /// AVX kernels on Intel Macs, plus vForce for whole-buffer
    /// transcendentals. Behind the `accelerate` feature, macOS only.
    /// Leads the gemm chains: it measured ahead of the Metal kernel
    /// at every size.
    Accelerate,
    /// Hand-written simdgroup-matrix GPU kernels for large `f32`
    /// products and maps (Metal has no `f64`), compiled from source
    /// at first use. Behind the `metal` feature, macOS only; serves
    /// what BLAS declines, and everything large in metal-only
    /// builds.
    Metal,
    /// cuBLAS on an NVIDIA GPU for large `f32`/`f64` products, its
    /// libraries bound at run time by `dlopen` so a machine without
    /// them declines instead of failing to build. Behind the `cuda`
    /// feature, Linux only; PCIe copies bound every task, so the
    /// threshold is high and residency stays future work.
    Cuda,
    /// The `matrixmultiply` crate's tuned CPU microkernels with
    /// runtime instruction-set dispatch (AVX-512F, AVX2+FMA, AVX,
    /// NEON), single-threaded. Behind the `simd` feature, every
    /// platform — the portable rung for Linux and everyone else,
    /// and mop-up behind the Apple backends on macOS.
    Simd,
    /// The crate's own fused kernels for composed formulas, elected
    /// onto plans at compile time and executing in-process through
    /// the payload seam — `windowed_product` today. Always compiled,
    /// always resident; the only implementer that can clear the
    /// bit-identity bar, because the oracle's bits live in this
    /// process.
    Fused,
    /// The StableHLO translation library: elected groups and leaf
    /// operations lower into a module a foreign runtime (XLA today)
    /// executes. Always compiled, always able to emit; running the
    /// module is that runtime's business, and its kernels answer
    /// under the envelope bar only.
    StableHlo,
}

impl Backend {
    /// Every implementer this crate version defines.
    pub const ALL: &'static [Backend] = &[
        Backend::Accelerate,
        Backend::Metal,
        Backend::Cuda,
        Backend::Simd,
        Backend::Fused,
        Backend::StableHlo,
    ];

    /// The coverage matrix: whether this implementer has a kernel
    /// for the formula, at what bar, for which precisions.
    ///
    /// Each row lives in its implementer's module and answers here
    /// through the `Implementer` contract; offer chains agree with
    /// the matrix by test, the plan's election reads
    /// [`Backend::Fused`]'s column, and emission reads
    /// [`Backend::StableHlo`]'s.
    pub fn coverage(self, formula: Formula) -> Cell {
        match self {
            Backend::Accelerate => Accelerate::coverage(formula),
            Backend::Metal => Metal::coverage(formula),
            Backend::Cuda => Cuda::coverage(formula),
            Backend::Simd => Simd::coverage(formula),
            Backend::Fused => Fused::coverage(formula),
            Backend::StableHlo => StableHlo::coverage(formula),
        }
    }

    /// Whether this implementer has a kernel for the formula that
    /// accepts jobs at this precision — designed coverage, the same
    /// answer in every build; [`status`](Backend::status) answers
    /// the orthogonal availability question.
    pub fn serves(self, formula: Formula, precision: Precision) -> bool {
        self.coverage(formula).admits(precision)
    }

    /// How this implementer's kernels are reached.
    pub fn dispatch(self) -> Dispatch {
        match self {
            Backend::Accelerate => Accelerate::DISPATCH,
            Backend::Metal => Metal::DISPATCH,
            Backend::Cuda => Cuda::DISPATCH,
            Backend::Simd => Simd::DISPATCH,
            Backend::Fused => Fused::DISPATCH,
            Backend::StableHlo => StableHlo::DISPATCH,
        }
    }

    /// Reports whether this implementer is in this build at all:
    /// the build-time half of [`status`](Backend::status), with no
    /// lazy setup and no device probe.
    ///
    /// Plans key elections on this answer, never on `status`, so a
    /// plan's shape depends only on the binary, not on which device
    /// happens to be plugged in.
    pub fn compiled(self) -> bool {
        match self {
            Backend::Accelerate => Accelerate::compiled(),
            Backend::Metal => Metal::compiled(),
            Backend::Cuda => Cuda::compiled(),
            Backend::Simd => Simd::compiled(),
            Backend::Fused => Fused::compiled(),
            Backend::StableHlo => StableHlo::compiled(),
        }
    }

    /// Reports whether this implementer would accept work in this
    /// build on this machine, forcing its lazy setup if it has one.
    ///
    /// `NotCompiled` is an ordinary answer, which is what lets a
    /// build without the feature ask the question; a loud program
    /// asserts readiness at startup with
    /// `Backend::Accelerate.status().expect(..)`.
    pub fn status(self) -> Result<(), BackendUnavailable> {
        match self {
            Backend::Accelerate => Accelerate::status(),
            Backend::Metal => Metal::status(),
            Backend::Cuda => Cuda::status(),
            Backend::Simd => Simd::status(),
            Backend::Fused => Fused::status(),
            Backend::StableHlo => StableHlo::status(),
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
