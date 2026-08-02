//! The Accelerate backend: dense GEMM on Apple's matrix units.
//!
//! One `cblas_sgemm`/`cblas_dgemm` call per task, executing on the
//! AMX/SME coprocessor on Apple Silicon (AVX kernels on Intel Macs)
//! with function-call latency — no device, no queue, no state. The
//! module is a pure function from task to product: classification of
//! the task's strides into BLAS transpose flags plus leading
//! dimensions, one foreign call, done. Tasks the mapping cannot
//! express (a stride-0 broadcast, dimensions beyond `i32`) and tasks
//! below the profitability threshold decline to the built-in paths.
//!
//! This is the crate's only `unsafe` code; the module is
//! scope-allowed under the crate-wide `deny(unsafe_code)`.

use crate::GemmTask;

// Row-major CBLAS constants.
const ROW_MAJOR: i32 = 101;
const NO_TRANSPOSE: i32 = 111;
const TRANSPOSE: i32 = 112;

/// Below this many floating-point operations (`2 * m * n * k`) the
/// built-in slice path wins on latency alone; the crossover sits
/// around n = 16 square and everything real is far above it.
const FLOP_THRESHOLD: usize = 1 << 13;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sgemm(
        order: i32,
        transpose_a: i32,
        transpose_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        leading_a: i32,
        b: *const f32,
        leading_b: i32,
        beta: f32,
        c: *mut f32,
        leading_c: i32,
    );
    fn cblas_dgemm(
        order: i32,
        transpose_a: i32,
        transpose_b: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        leading_a: i32,
        b: *const f64,
        leading_b: i32,
        beta: f64,
        c: *mut f64,
        leading_c: i32,
    );
}

/// One cblas-ready operand: the transpose flag and the leading
/// dimension.
struct CblasOperand {
    transpose: i32,
    leading: i32,
}

/// It classifies an operand's strides into cblas form, or declines:
/// `None` for patterns BLAS cannot express (a stride-0 broadcast
/// axis of extent above one) and for dimensions beyond `i32`.
///
/// A unit column stride is `NoTrans` with the row stride as the
/// leading dimension; a unit row stride is `Trans` with the column
/// stride leading. An extent-1 axis leaves its stride unused, so a
/// degenerate leading dimension is replaced by the smallest value
/// cblas accepts rather than declined.
fn classify(strides: [usize; 2], rows: usize, columns: usize) -> Option<CblasOperand> {
    if strides[1] == 1 {
        let leading = if rows == 1 { columns } else { strides[0] };
        if leading < columns {
            return None;
        }
        return Some(CblasOperand {
            transpose: NO_TRANSPOSE,
            leading: i32::try_from(leading).ok()?,
        });
    }
    if strides[0] == 1 {
        let leading = if columns == 1 { rows } else { strides[1] };
        if leading < rows {
            return None;
        }
        return Some(CblasOperand {
            transpose: TRANSPOSE,
            leading: i32::try_from(leading).ok()?,
        });
    }
    None
}

/// It runs a `f32` task through `cblas_sgemm`, or declines with
/// `None` when the task is below the threshold or outside the
/// mapping.
pub(super) fn gemm_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    if flops(task.m(), task.n(), task.k()) < FLOP_THRESHOLD {
        return None;
    }
    executed_f32(task)
}

/// It runs a `f64` task through `cblas_dgemm`, with the same decline
/// rules as the `f32` twin.
pub(super) fn gemm_f64(task: &GemmTask<'_, f64>) -> Option<Vec<f64>> {
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

/// The `f32` call without the threshold gate, so tests can drive the
/// mapping over shapes of every size.
pub(super) fn executed_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    let a = classify(task.a_strides(), task.m(), task.k())?;
    let b = classify(task.b_strides(), task.k(), task.n())?;
    let m = i32::try_from(task.m()).ok()?;
    let n = i32::try_from(task.n()).ok()?;
    let k = i32::try_from(task.k()).ok()?;
    let mut product = vec![0.0_f32; task.m() * task.n()];
    // SAFETY: the operand pointers come from live slices whose spans
    // the `GemmTask` constructor validated against the dimensions
    // and strides; `classify` guarantees the leading dimensions
    // satisfy the cblas access-pattern contract, so every read is in
    // bounds; `product` is exclusively borrowed and sized `m * n`
    // with `leading_c = n`; with `beta = 0` cblas only writes `c`
    // and only reads `a` and `b`.
    unsafe {
        cblas_sgemm(
            ROW_MAJOR,
            a.transpose,
            b.transpose,
            m,
            n,
            k,
            1.0,
            task.a().as_ptr(),
            a.leading,
            task.b().as_ptr(),
            b.leading,
            0.0,
            product.as_mut_ptr(),
            n,
        );
    }
    Some(product)
}

/// The `f64` call without the threshold gate; see [`executed_f32`].
pub(super) fn executed_f64(task: &GemmTask<'_, f64>) -> Option<Vec<f64>> {
    let a = classify(task.a_strides(), task.m(), task.k())?;
    let b = classify(task.b_strides(), task.k(), task.n())?;
    let m = i32::try_from(task.m()).ok()?;
    let n = i32::try_from(task.n()).ok()?;
    let k = i32::try_from(task.k()).ok()?;
    let mut product = vec![0.0_f64; task.m() * task.n()];
    // SAFETY: identical to `executed_f32` — validated spans, checked
    // leading dimensions, an exclusive `m * n` output, and cblas's
    // read/write contract under `beta = 0`.
    unsafe {
        cblas_dgemm(
            ROW_MAJOR,
            a.transpose,
            b.transpose,
            m,
            n,
            k,
            1.0,
            task.a().as_ptr(),
            a.leading,
            task.b().as_ptr(),
            b.leading,
            0.0,
            product.as_mut_ptr(),
            n,
        );
    }
    Some(product)
}

#[cfg(test)]
#[path = "tests/accelerate_tests.rs"]
mod tests;
