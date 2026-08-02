//! The metal backend: very large dense `f32` GEMM on the GPU.
//!
//! Hand-written simdgroup-matrix kernels (no MPS, no vendor library),
//! compiled from source once at first use and held as pipeline
//! states; shared-mode buffers come from a size-classed pool, and
//! every call is synchronous — encode, commit, wait, read back. The
//! threshold is high on purpose: unified memory makes the copies
//! cheap, but a command buffer costs ~100 us to submit and wait, so
//! the GPU only wins products around 800-square `f32` and above,
//! where Accelerate's curve flattens. Everything smaller declines to
//! the rest of the chain. Metal has no `f64` at all.
//!
//! Any failure degrades to slow, never to wrong: a failed setup or a
//! runtime command-buffer error poisons the module into declining
//! forever, with the reason held for [`status`].

mod context;
mod gemm;
mod pool;

use std::sync::OnceLock;

use crate::GemmTask;
use crate::backend::BackendUnavailable;

use self::context::Context;

/// Below this many floating-point operations (`2 * m * n * k`) the
/// dispatch latency outweighs the GPU's edge over the slice path —
/// the comparison that matters, since Accelerate leads the chain
/// wherever it is compiled in. The crossover sits near 256-square.
const FLOP_THRESHOLD: usize = 1 << 25;

static CONTEXT: OnceLock<Result<Context, String>> = OnceLock::new();
static POISON: OnceLock<String> = OnceLock::new();

/// Returns the lazily built context, or why the backend declines.
fn context() -> Result<&'static Context, BackendUnavailable> {
    if let Some(reason) = POISON.get() {
        return Err(BackendUnavailable::Poisoned(reason.clone()));
    }
    match CONTEXT.get_or_init(Context::new) {
        Ok(context) => Ok(context),
        Err(reason) => Err(BackendUnavailable::Initialization(reason.clone())),
    }
}

/// It reports readiness, forcing the lazy setup: device, kernel
/// compilation, and pipeline states.
pub(super) fn status() -> Result<(), BackendUnavailable> {
    context().map(|_| ())
}

/// It runs an `f32` task on the GPU, or declines with `None`: below
/// the threshold, GEMV-shaped, beyond `u32` extents, or with the
/// module poisoned or unavailable.
pub(super) fn gemm_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    let flops = 2usize
        .saturating_mul(task.m())
        .saturating_mul(task.n())
        .saturating_mul(task.k());
    if flops < FLOP_THRESHOLD {
        return None;
    }
    // Matrix tiles have nothing to feed on a vector.
    if task.m() == 1 || task.n() == 1 {
        return None;
    }
    if !fits_u32(task) {
        return None;
    }
    let context = context().ok()?;
    match gemm::executed(context, task, gemm::Kernel::Tiled) {
        Ok(product) => Some(product),
        Err(reason) => {
            // A numerics library degrades to slow, never to wrong:
            // one runtime failure disables the backend for good.
            let _ = POISON.set(reason);
            None
        }
    }
}

/// Returns whether every extent and stride fits the kernels' `u32`
/// parameters.
fn fits_u32(task: &GemmTask<'_, f32>) -> bool {
    let limit = u32::MAX as usize;
    task.m() <= limit
        && task.n() <= limit
        && task.k() <= limit
        && task.a_strides().iter().all(|&stride| stride <= limit)
        && task.b_strides().iter().all(|&stride| stride <= limit)
}

#[cfg(test)]
#[path = "tests/metal_tests.rs"]
mod tests;
