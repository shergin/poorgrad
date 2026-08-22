use static_assertions::assert_impl_all;

use super::backend::Backend;

// Request-time thread-safety contract; the anchor rationale is
// documented in `network.rs`.
assert_impl_all!(Formula: Send, Sync);
assert_impl_all!(Precision: Send, Sync);

/// Every formula the acceleration stack knows by name: the one
/// vocabulary, leaf and composed entries together.
///
/// A leaf entry is a single payload task (`Gemm`, `Map`); a composed
/// entry is a graph shape discovery recognizes (`WindowProduct` and
/// its siblings). Each entry has up to two faces — a buffer task
/// offerable at run time and a graph shape electable at compile
/// time — and the [`coverage`](Backend::coverage) matrix answers for
/// every `(implementer, formula)` pair, so a new variant cannot
/// compile until its cells are declared. Like [`Backend`], the enum
/// exists in every build; interrogating the stack never needs a
/// `cfg`.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Formula {
    /// One dense matrix product, a [`GemmTask`](crate::GemmTask).
    Gemm,
    /// One whole-buffer elementwise transcendental, a
    /// [`MapOperation`](crate::MapOperation) over a slice.
    Map,
    /// The im2col chain feeding a rank-2 product: convolution.
    WindowProduct,
    /// The max-pool window fold ending in the facade squeeze.
    ReduceWindow,
    /// Batch normalization by the batch's own statistics.
    BatchNormTraining,
    /// Batch normalization by supplied statistics.
    BatchNormInference,
}

/// The seam's forwarding precisions: the element types that route
/// payload tasks to hardware kernels.
///
/// The set is closed on purpose and is not a payload enumeration:
/// payloads stay open through `Elementary`, and a payload without a
/// forwarding precision computes on the reference paths or rides
/// through one, as `bf16` rides through `f32` expansion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Precision {
    /// IEEE 754 single precision.
    F32,
    /// IEEE 754 double precision.
    F64,
}

impl Formula {
    /// Every formula this crate version defines.
    pub const ALL: &'static [Formula] = &[
        Formula::Gemm,
        Formula::Map,
        Formula::WindowProduct,
        Formula::ReduceWindow,
        Formula::BatchNormTraining,
        Formula::BatchNormInference,
    ];

    /// The offer chain for this formula's buffer tasks at one
    /// precision: every offer-dispatched backend with a kernel for
    /// it, hardware-greediest first.
    ///
    /// The order is a measured decision, declared here once and
    /// pinned by tests; membership agrees with the
    /// [`coverage`](Backend::coverage) matrix by test. Composed
    /// formulas answer the empty chain: their kernels are elected
    /// onto plans and into modules, never offered buffers — until
    /// one earns a task face, which arrives as a chain here and a
    /// task type beside it.
    pub const fn chain(self, precision: Precision) -> &'static [Backend] {
        match self {
            // Accelerate leads the gemm chains: the measured
            // crossover has AMX ahead of the current Metal kernel at
            // every size, so Metal serves what BLAS declines (stride
            // patterns like broadcasts) and metal-only builds. The
            // order flips back if the kernel ever earns it. Metal
            // has no `f64` at all, so the `f64` chain skips it.
            Formula::Gemm => match precision {
                Precision::F32 => &[
                    Backend::Accelerate,
                    Backend::Metal,
                    Backend::Cuda,
                    Backend::Simd,
                ],
                Precision::F64 => &[Backend::Accelerate, Backend::Cuda, Backend::Simd],
            },
            // Metal leads the map chain, the reverse of the gemm
            // order: the measured crossover has the GPU ahead of
            // vForce from 512k elements, and its size gate hands
            // everything smaller to Accelerate behind it. The CPU
            // rungs end the map chains early: `matrixmultiply` is
            // GEMM-only, and a cuda map would be PCIe-bound.
            Formula::Map => match precision {
                Precision::F32 => &[Backend::Metal, Backend::Accelerate],
                Precision::F64 => &[Backend::Accelerate],
            },
            // The first composed formula with a task face: vDSP takes
            // the whole normalization in one offer. The CPU rungs
            // stay out — the in-process composed fallback is already
            // the CPU implementation.
            Formula::BatchNormTraining => &[Backend::Accelerate],
            Formula::WindowProduct | Formula::ReduceWindow | Formula::BatchNormInference => &[],
        }
    }
}

impl Precision {
    /// Every forwarding precision this crate version defines.
    pub const ALL: &'static [Precision] = &[Precision::F32, Precision::F64];
}

#[cfg(test)]
#[path = "tests/formula_tests.rs"]
mod tests;
