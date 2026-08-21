//! The cuda backend: large dense GEMM on an NVIDIA GPU via cuBLAS.
//!
//! One `cublasSgemm`/`cublasDgemm` call per task under the
//! column-major swap, with pooled device buffers and synchronous
//! copies each way — the metal architecture on discrete memory. The
//! libraries (`libcudart`, `libcublas`) are bound at run time by
//! `dlopen`, never at link time, so the build succeeds on every
//! machine and a missing toolkit is a status answer rather than a
//! build failure. The threshold is high on purpose: PCIe copies
//! dominate every task, so the GPU only pays for products around
//! 200-square and above, and even there the arm is copy-bound —
//! residency is a future payload design, not this rung.
//!
//! Any failure degrades to slow, never to wrong: a failed setup or
//! a runtime error poisons the module into declining forever, with
//! the reason held for [`status`].

// The kernel submodules keep their historical homes one level up;
// `path` mounts them under this `cfg`-gated half.
#[path = "context.rs"]
mod context;
#[path = "gemm.rs"]
mod gemm;
#[path = "pool.rs"]
mod pool;

use std::sync::OnceLock;

use crate::GemmTask;
use crate::backend::BackendUnavailable;
use crate::backend::operand::{Operand, classify};

use self::context::{Context, SetupError};

/// Below this many floating-point operations (`2 * m * n * k`) the
/// PCIe copies can never pay against the CPU rungs; the reasoned
/// crossover sits near 200-square, and the constant is provisional
/// until the first hardware measurement.
const FLOP_THRESHOLD: usize = 1 << 24;

static CONTEXT: OnceLock<Result<Context, SetupError>> = OnceLock::new();
static POISON: OnceLock<String> = OnceLock::new();

/// Returns the one-time setup outcome, building it on the first
/// call.
///
/// The typed error stays inside the module so the tests can skip on
/// a missing library or device yet fail hard on every other setup
/// defect.
fn initialized() -> &'static Result<Context, SetupError> {
    CONTEXT.get_or_init(Context::new)
}

/// Returns the lazily built context, or why the backend declines.
fn context() -> Result<&'static Context, BackendUnavailable> {
    if let Some(reason) = POISON.get() {
        return Err(BackendUnavailable::Poisoned(reason.clone()));
    }
    match initialized() {
        Ok(context) => Ok(context),
        Err(error) => Err(BackendUnavailable::Initialization(error.to_string())),
    }
}

/// It reports readiness, forcing the lazy setup: library loading,
/// the device probe, and handle creation.
pub(super) fn status() -> Result<(), BackendUnavailable> {
    context().map(|_| ())
}

/// The declines that need no device: GEMV shapes, sub-threshold
/// sizes, unclassifiable strides, and dimensions beyond `i32` all
/// answer before the libraries are even opened.
fn eligible<Element>(task: &GemmTask<'_, Element>) -> Option<(Operand, Operand, i32, i32, i32)> {
    // Matrix offload has nothing to feed on a vector: copies alone
    // sink it.
    if task.m() == 1 || task.n() == 1 {
        return None;
    }
    let flops = 2usize
        .saturating_mul(task.m())
        .saturating_mul(task.n())
        .saturating_mul(task.k());
    if flops < FLOP_THRESHOLD {
        return None;
    }
    task.m().checked_mul(task.n())?;
    let a = classify(task.a_strides(), task.m(), task.k())?;
    let b = classify(task.b_strides(), task.k(), task.n())?;
    let m = i32::try_from(task.m()).ok()?;
    let n = i32::try_from(task.n()).ok()?;
    let k = i32::try_from(task.k()).ok()?;
    Some((a, b, m, n, k))
}

/// It runs an `f32` task on the GPU, or declines with `None`: below
/// the threshold, GEMV-shaped, outside the stride mapping, or with
/// the module poisoned or unavailable.
pub(crate) fn gemm_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    let (a, b, m, n, k) = eligible(task)?;
    let context = context().ok()?;
    match gemm::executed_f32(context, task, &a, &b, m, n, k) {
        Ok(product) => Some(product),
        Err(reason) => {
            // A numerics library degrades to slow, never to wrong:
            // one runtime failure disables the backend for good.
            let _ = POISON.set(reason);
            None
        }
    }
}

/// The `f64` twin of [`gemm_f32`]; GeForce-class cards run it at a
/// fraction of their `f32` rate, which the docs state rather than
/// hide.
pub(crate) fn gemm_f64(task: &GemmTask<'_, f64>) -> Option<Vec<f64>> {
    let (a, b, m, n, k) = eligible(task)?;
    let context = context().ok()?;
    match gemm::executed_f64(context, task, &a, &b, m, n, k) {
        Ok(product) => Some(product),
        Err(reason) => {
            let _ = POISON.set(reason);
            None
        }
    }
}

#[cfg(test)]
#[path = "tests/cuda_tests.rs"]
mod tests;
