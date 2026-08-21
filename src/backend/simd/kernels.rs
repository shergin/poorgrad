//! The simd backend: tuned CPU GEMM through `matrixmultiply`.
//!
//! One `sgemm`/`dgemm` call per task into hand-written microkernels
//! with runtime instruction-set dispatch (AVX-512F, AVX2+FMA, AVX,
//! NEON, wasm SIMD128) — no device, no state, no platform `cfg`: the
//! portable acceleration rung for Linux and everyone else, and the
//! only backend that is real on every OS. The crate's strided API
//! takes `GemmTask`'s row and column strides directly, so transposed
//! operands need no classification; only stride-0 broadcasts, tasks
//! whose offsets could overflow `isize`, and tasks below the
//! profitability threshold decline to the built-in paths.
//!
//! Single-threaded on purpose: the provider's `threading` feature
//! stays off, so there is no thread-count knob to perturb results
//! and no contention with user-level parallel runs.

use crate::GemmTask;

/// Below this many floating-point operations (`2 * m * n * k`) the
/// built-in slice path wins on latency alone. Provisionally the
/// accelerate value; retuned when the measurement plan runs.
const FLOP_THRESHOLD: usize = 1 << 13;

/// It runs a `f32` task through `matrixmultiply::sgemm`, or declines
/// with `None` when the task is below the threshold or outside the
/// mapping.
pub(crate) fn gemm_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    if flops(task.m(), task.n(), task.k()) < FLOP_THRESHOLD {
        return None;
    }
    executed_f32(task)
}

/// It runs a `f64` task through `matrixmultiply::dgemm`, with the
/// same decline rules as the `f32` twin.
pub(crate) fn gemm_f64(task: &GemmTask<'_, f64>) -> Option<Vec<f64>> {
    if flops(task.m(), task.n(), task.k()) < FLOP_THRESHOLD {
        return None;
    }
    executed_f64(task)
}

/// Returns the task's floating-point operation count, saturating —
/// a saturated count is enormous and therefore above any threshold.
fn flops(m: usize, n: usize, k: usize) -> usize {
    2usize.saturating_mul(m).saturating_mul(n).saturating_mul(k)
}

/// It converts an operand's strides for the kernel, or declines:
/// `None` for stride-0 broadcasts (the conservative mirror of the
/// cblas arm's decline) and wherever the furthest element offset —
/// which bounds every offset the kernel computes — cannot be an
/// `isize`.
fn strides_for(strides: [usize; 2], rows: usize, columns: usize) -> Option<[isize; 2]> {
    if strides[0] == 0 || strides[1] == 0 {
        return None;
    }
    let row_stride = isize::try_from(strides[0]).ok()?;
    let column_stride = isize::try_from(strides[1]).ok()?;
    let furthest = (rows - 1)
        .checked_mul(strides[0])?
        .checked_add((columns - 1).checked_mul(strides[1])?)?;
    isize::try_from(furthest).ok()?;
    Some([row_stride, column_stride])
}

/// The `f32` call without the threshold gate, so tests can drive the
/// mapping over shapes of every size.
pub(super) fn executed_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    let a = strides_for(task.a_strides(), task.m(), task.k())?;
    let b = strides_for(task.b_strides(), task.k(), task.n())?;
    let volume = task.m().checked_mul(task.n())?;
    isize::try_from(volume).ok()?;
    let mut product = vec![0.0_f32; volume];
    // SAFETY: the operand pointers come from live slices whose spans
    // the `GemmTask` constructor validated against the dimensions
    // and strides; `strides_for` proves the furthest offset of each
    // operand fits `isize`, bounding the kernel's offset arithmetic;
    // `product` is exclusively borrowed at `m * n` (proven to fit
    // `isize`) with the dense row-major strides `[n, 1]`; with
    // `beta = 0` the kernel only writes `c` and only reads `a` and
    // `b`.
    unsafe {
        matrixmultiply::sgemm(
            task.m(),
            task.k(),
            task.n(),
            1.0,
            task.a().as_ptr(),
            a[0],
            a[1],
            task.b().as_ptr(),
            b[0],
            b[1],
            0.0,
            product.as_mut_ptr(),
            task.n() as isize,
            1,
        );
    }
    Some(product)
}

/// The `f64` call without the threshold gate; see [`executed_f32`].
pub(super) fn executed_f64(task: &GemmTask<'_, f64>) -> Option<Vec<f64>> {
    let a = strides_for(task.a_strides(), task.m(), task.k())?;
    let b = strides_for(task.b_strides(), task.k(), task.n())?;
    let volume = task.m().checked_mul(task.n())?;
    isize::try_from(volume).ok()?;
    let mut product = vec![0.0_f64; volume];
    // SAFETY: identical to `executed_f32` — validated spans, checked
    // offsets, an exclusive `m * n` output, and the kernel's
    // read/write contract under `beta = 0`.
    unsafe {
        matrixmultiply::dgemm(
            task.m(),
            task.k(),
            task.n(),
            1.0,
            task.a().as_ptr(),
            a[0],
            a[1],
            task.b().as_ptr(),
            b[0],
            b[1],
            0.0,
            product.as_mut_ptr(),
            task.n() as isize,
            1,
        );
    }
    Some(product)
}

#[cfg(test)]
#[path = "../tests/simd_tests.rs"]
mod tests;
