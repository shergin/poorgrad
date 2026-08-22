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
//! This is the crate's only `unsafe` code in an accelerate-only
//! build (every other backend feature carries its own); each
//! backend's `kernels` submodule is scope-allowed under the
//! crate-wide `deny(unsafe_code)`, while the descriptor half stays
//! outside the allow.

use crate::backend::operand::{Operand, classify};
use crate::{BatchNormTask, GemmTask, MapOperation, Normalized};

// Row-major CBLAS constants.
const ROW_MAJOR: i32 = 101;
const NO_TRANSPOSE: i32 = 111;
const TRANSPOSE: i32 = 112;

/// Returns the cblas transpose constant for a classified operand.
fn transpose(operand: &Operand) -> i32 {
    if operand.transposed {
        TRANSPOSE
    } else {
        NO_TRANSPOSE
    }
}

/// Below this many floating-point operations (`2 * m * n * k`) the
/// built-in slice path wins on latency alone; the crossover sits
/// around n = 16 square and everything real is far above it.
const FLOP_THRESHOLD: usize = 1 << 13;

/// Below this many elements a vForce call's setup outweighs the
/// scalar loop; the crossover is small and flat, so the constant is
/// conservative rather than tuned.
const MAP_THRESHOLD: usize = 1 << 7;

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
    // vForce: vectorized transcendentals over whole buffers, the
    // library form of the loops libm calls keep scalar.
    fn vvexpf(mapped: *mut f32, elements: *const f32, count: *const i32);
    fn vvlogf(mapped: *mut f32, elements: *const f32, count: *const i32);
    fn vvsqrtf(mapped: *mut f32, elements: *const f32, count: *const i32);
    fn vvtanhf(mapped: *mut f32, elements: *const f32, count: *const i32);
    fn vvexp(mapped: *mut f64, elements: *const f64, count: *const i32);
    fn vvlog(mapped: *mut f64, elements: *const f64, count: *const i32);
    fn vvsqrt(mapped: *mut f64, elements: *const f64, count: *const i32);
    fn vvtanh(mapped: *mut f64, elements: *const f64, count: *const i32);
    // vDSP: strided vector statistics and the fused multiply-add
    // behind the batch-normalization kernel. Strides are in
    // elements; counts are element counts.
    fn vDSP_meanv(elements: *const f32, stride: isize, mean: *mut f32, count: usize);
    fn vDSP_measqv(elements: *const f32, stride: isize, mean_square: *mut f32, count: usize);
    fn vDSP_vsadd(
        elements: *const f32,
        stride: isize,
        addend: *const f32,
        shifted: *mut f32,
        shifted_stride: isize,
        count: usize,
    );
    fn vDSP_vsmsa(
        elements: *const f32,
        stride: isize,
        multiplier: *const f32,
        addend: *const f32,
        transformed: *mut f32,
        transformed_stride: isize,
        count: usize,
    );
    fn vDSP_meanvD(elements: *const f64, stride: isize, mean: *mut f64, count: usize);
    fn vDSP_measqvD(elements: *const f64, stride: isize, mean_square: *mut f64, count: usize);
    fn vDSP_vsaddD(
        elements: *const f64,
        stride: isize,
        addend: *const f64,
        shifted: *mut f64,
        shifted_stride: isize,
        count: usize,
    );
    fn vDSP_vsmsaD(
        elements: *const f64,
        stride: isize,
        multiplier: *const f64,
        addend: *const f64,
        transformed: *mut f64,
        transformed_stride: isize,
        count: usize,
    );
}

/// Below this many elements the per-feature vDSP call overhead
/// outweighs the composed path; the constant is provisional until
/// the measurement plan runs.
const BATCH_NORM_THRESHOLD: usize = 1 << 12;

/// It runs a whole training-mode batch normalization through vDSP —
/// per feature: a strided mean, the centered mean of squares (the
/// biased variance), and the affine as one fused multiply-add — or
/// declines with `None` below the threshold.
pub(crate) fn batch_norm_f32(task: &BatchNormTask<'_, f32>) -> Option<Normalized<f32>> {
    let (batch, features) = (task.batch(), task.features());
    if batch * features < BATCH_NORM_THRESHOLD {
        return None;
    }
    let input = task.input();
    let scale = task.scale();
    let shift = task.shift();
    let epsilon = *task.epsilon();
    let stride = features as isize;
    let mut output = vec![0.0_f32; batch * features];
    let mut mean = vec![0.0_f32; features];
    let mut variance = vec![0.0_f32; features];
    let mut centered = vec![0.0_f32; batch];
    for feature in 0..features {
        let column = &input[feature..];
        // SAFETY: `column` starts at this feature's first element of
        // a slice the task constructor validated at
        // `batch * features`, so every strided access
        // `column[i * features]` for `i < batch` stays in bounds;
        // `centered` and the output column are exclusively borrowed
        // at the lengths the calls write; vDSP reads its inputs and
        // count while writing only its output argument.
        unsafe {
            vDSP_meanv(column.as_ptr(), stride, &mut mean[feature], batch);
            let negative = -mean[feature];
            vDSP_vsadd(
                column.as_ptr(),
                stride,
                &negative,
                centered.as_mut_ptr(),
                1,
                batch,
            );
            vDSP_measqv(centered.as_ptr(), 1, &mut variance[feature], batch);
            let normalizer = scale[feature] / (variance[feature] + epsilon).sqrt();
            let offset = shift[feature] - mean[feature] * normalizer;
            vDSP_vsmsa(
                column.as_ptr(),
                stride,
                &normalizer,
                &offset,
                output[feature..].as_mut_ptr(),
                stride,
                batch,
            );
        }
    }
    Some(Normalized {
        output,
        mean,
        variance,
    })
}

/// The `f64` twin of [`batch_norm_f32`].
pub(crate) fn batch_norm_f64(task: &BatchNormTask<'_, f64>) -> Option<Normalized<f64>> {
    let (batch, features) = (task.batch(), task.features());
    if batch * features < BATCH_NORM_THRESHOLD {
        return None;
    }
    let input = task.input();
    let scale = task.scale();
    let shift = task.shift();
    let epsilon = *task.epsilon();
    let stride = features as isize;
    let mut output = vec![0.0_f64; batch * features];
    let mut mean = vec![0.0_f64; features];
    let mut variance = vec![0.0_f64; features];
    let mut centered = vec![0.0_f64; batch];
    for feature in 0..features {
        let column = &input[feature..];
        // SAFETY: identical to `batch_norm_f32` — validated span,
        // in-bounds strided accesses, exclusive outputs, and vDSP's
        // read/write contract.
        unsafe {
            vDSP_meanvD(column.as_ptr(), stride, &mut mean[feature], batch);
            let negative = -mean[feature];
            vDSP_vsaddD(
                column.as_ptr(),
                stride,
                &negative,
                centered.as_mut_ptr(),
                1,
                batch,
            );
            vDSP_measqvD(centered.as_ptr(), 1, &mut variance[feature], batch);
            let normalizer = scale[feature] / (variance[feature] + epsilon).sqrt();
            let offset = shift[feature] - mean[feature] * normalizer;
            vDSP_vsmsaD(
                column.as_ptr(),
                stride,
                &normalizer,
                &offset,
                output[feature..].as_mut_ptr(),
                stride,
                batch,
            );
        }
    }
    Some(Normalized {
        output,
        mean,
        variance,
    })
}

/// It runs a `f32` task through `cblas_sgemm`, or declines with
/// `None` when the task is below the threshold or outside the
/// mapping.
pub(crate) fn gemm_f32(task: &GemmTask<'_, f32>) -> Option<Vec<f32>> {
    if flops(task.m(), task.n(), task.k()) < FLOP_THRESHOLD {
        return None;
    }
    executed_f32(task)
}

/// It runs a `f64` task through `cblas_dgemm`, with the same decline
/// rules as the `f32` twin.
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
            transpose(&a),
            transpose(&b),
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

/// It maps one transcendental over an `f32` buffer through vForce,
/// declining buffers too small to pay the call or too long for the
/// interface's `i32` count.
pub(crate) fn map_f32(operation: MapOperation, elements: &[f32]) -> Option<Vec<f32>> {
    if elements.len() < MAP_THRESHOLD {
        return None;
    }
    let count = i32::try_from(elements.len()).ok()?;
    let mut mapped = vec![0.0_f32; elements.len()];
    // SAFETY: both pointers address live buffers of exactly `count`
    // elements — `mapped` exclusively — and vForce reads the input
    // and count while writing only the output.
    unsafe {
        match operation {
            MapOperation::Exp => vvexpf(mapped.as_mut_ptr(), elements.as_ptr(), &count),
            MapOperation::Ln => vvlogf(mapped.as_mut_ptr(), elements.as_ptr(), &count),
            MapOperation::Sqrt => vvsqrtf(mapped.as_mut_ptr(), elements.as_ptr(), &count),
            MapOperation::Tanh => vvtanhf(mapped.as_mut_ptr(), elements.as_ptr(), &count),
        }
    }
    Some(mapped)
}

/// The `f64` twin of [`map_f32`].
pub(crate) fn map_f64(operation: MapOperation, elements: &[f64]) -> Option<Vec<f64>> {
    if elements.len() < MAP_THRESHOLD {
        return None;
    }
    let count = i32::try_from(elements.len()).ok()?;
    let mut mapped = vec![0.0_f64; elements.len()];
    // SAFETY: identical to `map_f32` — live buffers of `count`
    // elements, exclusive output, read-only input.
    unsafe {
        match operation {
            MapOperation::Exp => vvexp(mapped.as_mut_ptr(), elements.as_ptr(), &count),
            MapOperation::Ln => vvlog(mapped.as_mut_ptr(), elements.as_ptr(), &count),
            MapOperation::Sqrt => vvsqrt(mapped.as_mut_ptr(), elements.as_ptr(), &count),
            MapOperation::Tanh => vvtanh(mapped.as_mut_ptr(), elements.as_ptr(), &count),
        }
    }
    Some(mapped)
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
            transpose(&a),
            transpose(&b),
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
