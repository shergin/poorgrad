use static_assertions::assert_impl_all;

use super::backend::Backend;

// Request-time thread-safety contract; the anchor rationale is
// documented in `network.rs`.
assert_impl_all!(TaskKind: Send, Sync);

/// The kinds of work the backend chain can be offered: the seam's
/// two hooks crossed with the two element types that forward to it.
///
/// The enum names the chain's task vocabulary, so coverage and
/// priority are declared data instead of ladder control flow: each
/// kind's [`chain`](TaskKind::chain) is the offer order, membership
/// in it is the coverage claim [`Backend::serves`] answers, and a
/// future kind (a batched product, a convolution) arrives as a new
/// variant. Like [`Backend`], the enum exists in every build, so
/// interrogating the chain never needs a `cfg`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    /// One dense `f32` matrix product, a [`GemmTask`](crate::GemmTask).
    GemmF32,
    /// One dense `f64` matrix product.
    GemmF64,
    /// One whole-buffer `f32` transcendental, a
    /// [`MapOperation`](crate::MapOperation) over a slice.
    MapF32,
    /// One whole-buffer `f64` transcendental.
    MapF64,
}

impl TaskKind {
    /// Every task kind this crate version defines.
    pub const ALL: &'static [TaskKind] = &[
        TaskKind::GemmF32,
        TaskKind::GemmF64,
        TaskKind::MapF32,
        TaskKind::MapF64,
    ];

    /// The chain for this kind: every backend with a kernel for it,
    /// in offer order, hardware-greediest first.
    ///
    /// The order is a measured decision, declared here once and
    /// pinned by tests. Whether a member is compiled into a build is
    /// the orthogonal [`Backend::status`] answer; dispatch offers
    /// the task down this chain and a member missing from the build
    /// simply declines.
    pub const fn chain(self) -> &'static [Backend] {
        match self {
            // Accelerate leads the gemm chains: the measured
            // crossover has AMX ahead of the current Metal kernel at
            // every size, so Metal serves what BLAS declines (stride
            // patterns like broadcasts) and metal-only builds. The
            // order flips back if the kernel ever earns it.
            TaskKind::GemmF32 => &[
                Backend::Accelerate,
                Backend::Metal,
                Backend::Cuda,
                Backend::Simd,
            ],
            // Metal has no `f64` at all, so the `f64` chain skips it.
            TaskKind::GemmF64 => &[Backend::Accelerate, Backend::Cuda, Backend::Simd],
            // Metal leads the map chain, the reverse of the gemm
            // order: the measured crossover has the GPU ahead of
            // vForce from 512k elements, and its size gate hands
            // everything smaller to Accelerate behind it. The CPU
            // rungs end the map chains early: `matrixmultiply` is
            // GEMM-only, and a cuda map would be PCIe-bound.
            TaskKind::MapF32 => &[Backend::Metal, Backend::Accelerate],
            TaskKind::MapF64 => &[Backend::Accelerate],
        }
    }
}

#[cfg(test)]
#[path = "tests/task_tests.rs"]
mod tests;
