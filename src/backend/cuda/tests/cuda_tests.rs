use std::ops::{Add, Mul};

use crate::backend::operand::classify;
use crate::{Differentiable, GemmTask, Network, Shape, Tensor, init};

use super::context::{Context, SetupError};
use super::{gemm, gemm_f32, gemm_f64, initialized};

/// Returns the context, or `None` on machines without the NVIDIA
/// libraries or without a device — every CI runner today — where
/// the GPU tests skip rather than fail, honoring the backend's own
/// degrade-to-slow contract. Any other setup failure — a missing
/// symbol, a failed handle — stays a hard error, so a broken
/// backend can never turn the grid green by skipping it.
fn device() -> Option<&'static Context> {
    match initialized() {
        Ok(context) => Some(context),
        Err(SetupError::NoLibrary(name)) => {
            eprintln!("skipping: `{name}` is not available");
            None
        }
        Err(SetupError::NoDevice) => {
            eprintln!("skipping: no CUDA device");
            None
        }
        Err(SetupError::Failed(reason)) => panic!("CUDA setup failed: {reason}"),
    }
}

/// Runs one `f64` task through the gemm call without the size
/// gates, so the grid can drive shapes of every size.
fn executed_f64(context: &Context, task: &GemmTask<'_, f64>) -> Vec<f64> {
    let a = classify(task.a_strides(), task.m(), task.k()).expect("a classifiable operand");
    let b = classify(task.b_strides(), task.k(), task.n()).expect("a classifiable operand");
    let m = i32::try_from(task.m()).expect("a small test dimension");
    let n = i32::try_from(task.n()).expect("a small test dimension");
    let k = i32::try_from(task.k()).expect("a small test dimension");
    gemm::executed_f64(context, task, &a, &b, m, n, k).expect("the gemm call succeeds")
}

/// The `f32` twin of [`executed_f64`].
fn executed_f32(context: &Context, task: &GemmTask<'_, f32>) -> Vec<f32> {
    let a = classify(task.a_strides(), task.m(), task.k()).expect("a classifiable operand");
    let b = classify(task.b_strides(), task.k(), task.n()).expect("a classifiable operand");
    let m = i32::try_from(task.m()).expect("a small test dimension");
    let n = i32::try_from(task.n()).expect("a small test dimension");
    let k = i32::try_from(task.k()).expect("a small test dimension");
    gemm::executed_f32(context, task, &a, &b, m, n, k).expect("the gemm call succeeds")
}

/// Computes the product of two row-major matrices in the logical
/// path's accumulation order: the reference every case compares to,
/// within a tolerance, since cuBLAS sums in its own order.
fn reference<Element: Copy + Add<Output = Element> + Mul<Output = Element>>(
    a: &[Element],
    b: &[Element],
    m: usize,
    k: usize,
    n: usize,
) -> Vec<Element> {
    let mut elements = Vec::with_capacity(m * n);
    for row in 0..m {
        for column in 0..n {
            let mut total = a[row * k] * b[column];
            for step in 1..k {
                total = total + a[row * k + step] * b[step * n + column];
            }
            elements.push(total);
        }
    }
    elements
}

/// Returns distinct, sign-varied row-major values for a matrix.
fn varied(rows: usize, columns: usize, seed: i64) -> Vec<f64> {
    (0..(rows * columns) as i64)
        .map(|index| ((index * 7 + seed * 13) % 23 - 11) as f64 / 4.0)
        .collect()
}

/// A buffer holding a logical matrix under one stride form.
struct Form {
    buffer: Vec<f64>,
    strides: [usize; 2],
}

/// One stride-form builder of the grid.
type FormBuilder = fn(&[f64], usize, usize) -> Form;

/// Returns the logical matrix stored contiguously.
fn contiguous(logical: &[f64], _rows: usize, columns: usize) -> Form {
    Form {
        buffer: logical.to_vec(),
        strides: [columns, 1],
    }
}

/// Returns the logical matrix stored as its transpose, read back
/// through a transposed view's strides.
fn transposed(logical: &[f64], rows: usize, columns: usize) -> Form {
    let mut buffer = vec![0.0; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            buffer[column * rows + row] = logical[row * columns + column];
        }
    }
    Form {
        buffer,
        strides: [1, rows],
    }
}

/// Returns the logical matrix stored with three padding elements per
/// row: a narrowed window's wide leading dimension.
fn narrowed(logical: &[f64], rows: usize, columns: usize) -> Form {
    let padded = columns + 3;
    let mut buffer = vec![0.0; rows * padded];
    for row in 0..rows {
        buffer[row * padded..row * padded + columns]
            .copy_from_slice(&logical[row * columns..(row + 1) * columns]);
    }
    Form {
        buffer,
        strides: [padded, 1],
    }
}

/// Asserts elementwise closeness under the k-scaled tolerance of the
/// design note (cuBLAS reorders sums; it must not change them).
fn assert_close_f64(actual: &[f64], expected: &[f64], k: usize) {
    let tolerance = 8.0 * f64::EPSILON * (k as f64).sqrt();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "{actual} differs from {expected} beyond tolerance (k = {k})"
        );
    }
}

/// The `f32` twin of [`assert_close_f64`].
fn assert_close_f32(actual: &[f32], expected: &[f32], k: usize) {
    let tolerance = 8.0 * f32::EPSILON * (k as f32).sqrt();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "{actual} differs from {expected} beyond tolerance (k = {k})"
        );
    }
}

/// The shape grid: squares, rectangles, and primes. GEMV shapes are
/// absent on purpose — the arm declines them by design.
const SHAPES: [(usize, usize, usize); 7] = [
    (2, 3, 4),
    (5, 8, 13),
    (16, 16, 16),
    (63, 64, 65),
    (100, 127, 3),
    (128, 100, 64),
    (2, 64, 128),
];

#[test]
fn every_stride_form_matches_the_reference_f64() {
    let Some(context) = device() else { return };
    let forms: [FormBuilder; 3] = [contiguous, transposed, narrowed];
    for (m, k, n) in SHAPES {
        let left = varied(m, k, 1);
        let right = varied(k, n, 2);
        let expected = reference(&left, &right, m, k, n);
        for left_form in forms {
            for right_form in forms {
                let a = left_form(&left, m, k);
                let b = right_form(&right, k, n);
                let task = GemmTask::new(&a.buffer, a.strides, &b.buffer, b.strides, m, k, n);
                assert_close_f64(&executed_f64(context, &task), &expected, k);
            }
        }
    }
}

#[test]
fn every_stride_form_matches_the_reference_f32() {
    let Some(context) = device() else { return };
    for (m, k, n) in SHAPES {
        let left: Vec<f32> = varied(m, k, 3).iter().map(|&value| value as f32).collect();
        let right: Vec<f32> = varied(k, n, 4).iter().map(|&value| value as f32).collect();
        let expected = reference(&left, &right, m, k, n);
        let task = GemmTask::new(&left, [k, 1], &right, [n, 1], m, k, n);
        assert_close_f32(&executed_f32(context, &task), &expected, k);

        let mut transposed_right = vec![0.0_f32; k * n];
        for row in 0..k {
            for column in 0..n {
                transposed_right[column * k + row] = right[row * n + column];
            }
        }
        let task = GemmTask::new(&left, [k, 1], &transposed_right, [1, k], m, k, n);
        assert_close_f32(&executed_f32(context, &task), &expected, k);
    }
}

#[test]
fn gemv_shapes_decline_before_any_device_work() {
    // These declines answer inside `eligible`, before the libraries
    // are even opened, so they hold on machines with no GPU at all.
    let row = varied(1, 8, 1);
    let right = varied(8, 4, 2);
    let task = GemmTask::new(&row, [8, 1], &right, [4, 1], 1, 8, 4);
    assert_eq!(gemm_f64(&task), None);

    let left = varied(4, 8, 3);
    let column = varied(8, 1, 4);
    let task = GemmTask::new(&left, [8, 1], &column, [1, 1], 4, 8, 1);
    assert_eq!(gemm_f64(&task), None);
}

#[test]
fn the_threshold_declines_small_tasks() {
    let left = varied(16, 16, 1);
    let right = varied(16, 16, 2);
    let task = GemmTask::new(&left, [16, 1], &right, [16, 1], 16, 16, 16);
    let left32: Vec<f32> = left.iter().map(|&value| value as f32).collect();
    let right32: Vec<f32> = right.iter().map(|&value| value as f32).collect();
    let task32 = GemmTask::new(&left32, [16, 1], &right32, [16, 1], 16, 16, 16);
    assert_eq!(gemm_f64(&task), None);
    assert_eq!(gemm_f32(&task32), None);
}

#[test]
fn broadcast_strides_decline_before_any_device_work() {
    // Above the flop threshold so the stride classification is what
    // declines, still with a one-row buffer: the stride-0 span only
    // covers `k` elements.
    let row = varied(1, 256, 1);
    let right = varied(256, 256, 2);
    let task = GemmTask::new(&row, [0, 1], &right, [256, 1], 256, 256, 256);
    assert_eq!(gemm_f64(&task), None);
}

#[test]
fn repeated_products_answer_bitwise_identically() {
    // cuBLAS selects a deterministic kernel for a fixed shape on a
    // fixed device and library version; per the house rule its
    // run-to-run determinism is verified, not assumed. The observed
    // answer is recorded in notes/cuda-backend.md.
    let Some(context) = device() else { return };
    let size = 1024;
    let left = varied(size, size, 5);
    let right = varied(size, size, 6);
    let task = GemmTask::new(&left, [size, 1], &right, [size, 1], size, size, size);
    let first = executed_f64(context, &task);
    let second = executed_f64(context, &task);
    let first_bits: Vec<u64> = first.iter().map(|value| value.to_bits()).collect();
    let second_bits: Vec<u64> = second.iter().map(|value| value.to_bits()).collect();
    assert_eq!(first_bits, second_bits);
}

#[test]
fn training_runs_through_the_backend_end_to_end() {
    // A batch-512 regression whose hidden products sit above the
    // flop threshold, so forward and backward (transposed views
    // included) route through cuBLAS against the real tape; the
    // output layer is GEMV-shaped and exercises the mixed-routing
    // fallback in the same run.
    let Some(_context) = device() else { return };
    let network = Network::new();
    let inputs = network.input(Tensor::filled([512, 256], 0.5_f64));
    let targets = network.input(Tensor::filled([512, 1], 1.0_f64));
    let mut initializer = init::uniform(11, 0.05);
    let weights = network.parameter(initializer(&Shape::new([256, 128])));
    let output_weights = network.parameter(initializer(&Shape::new([128, 1])));
    let hidden = inputs.matmul(weights).tanh();
    let prediction = hidden.matmul(output_weights);
    let error = prediction - targets;
    let loss = (error * error).sum();
    let loss_symbol = loss.symbol();

    let mut network = network;
    let mut first_loss = None;
    let mut last_loss = f64::INFINITY;
    for _ in 0..30 {
        let loss_value = network.resolve(loss_symbol);
        let evaluation = network.forward();
        last_loss = evaluation.of(loss_value).to_vec()[0];
        first_loss.get_or_insert(last_loss);
        let gradients = evaluation.backward(loss_value);
        network = network.update(&gradients, |weight, gradient| {
            weight.clone() - Tensor::filled(gradient.shape(), 0.0002) * gradient.clone()
        });
    }
    let first_loss = first_loss.expect("the loop ran");
    assert!(
        last_loss.is_finite() && last_loss < first_loss * 0.5,
        "training through the backend did not converge: {first_loss} -> {last_loss}"
    );
}
