use std::ops::{Add, Mul};

use crate::{
    BatchNormTask, Differentiable, Elementary, GemmTask, MapOperation, Shape, Tape, Tensor,
    Tensorial, init,
};

use super::{
    batch_norm_f32, batch_norm_f64, executed_f32, executed_f64, gemm_f32, map_f32, map_f64,
};

/// Computes the product of two row-major matrices in the logical
/// path's accumulation order: the reference every case compares to,
/// within a tolerance, since Accelerate sums in its own order.
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
/// design note (Accelerate reorders sums; it must not change them).
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

/// The shape grid: squares, rectangles, primes, and GEMV shapes.
const SHAPES: [(usize, usize, usize); 9] = [
    (1, 1, 1),
    (2, 3, 4),
    (5, 8, 13),
    (16, 16, 16),
    (63, 64, 65),
    (100, 127, 3),
    (128, 100, 64),
    (1, 64, 128),
    (64, 128, 1),
];

#[test]
fn every_stride_form_matches_the_reference_f64() {
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
                let product = executed_f64(&task).expect("the mapping accepts this form");
                assert_close_f64(&product, &expected, k);
            }
        }
    }
}

#[test]
fn every_stride_form_matches_the_reference_f32() {
    for (m, k, n) in SHAPES {
        let left: Vec<f32> = varied(m, k, 3).iter().map(|&value| value as f32).collect();
        let right: Vec<f32> = varied(k, n, 4).iter().map(|&value| value as f32).collect();
        let expected = reference(&left, &right, m, k, n);
        let task = GemmTask::new(&left, [k, 1], &right, [n, 1], m, k, n);
        let product = executed_f32(&task).expect("the mapping accepts contiguous operands");
        assert_close_f32(&product, &expected, k);

        let mut transposed_right = vec![0.0_f32; k * n];
        for row in 0..k {
            for column in 0..n {
                transposed_right[column * k + row] = right[row * n + column];
            }
        }
        let task = GemmTask::new(&left, [k, 1], &transposed_right, [1, k], m, k, n);
        let product = executed_f32(&task).expect("the mapping accepts a transposed operand");
        assert_close_f32(&product, &expected, k);
    }
}

#[test]
fn broadcast_strides_decline_to_the_built_in_paths() {
    let row = varied(1, 8, 1);
    let right = varied(8, 4, 2);
    // A stride-0 row axis is a pattern BLAS cannot express.
    let task = GemmTask::new(&row, [0, 1], &right, [4, 1], 3, 8, 4);
    assert_eq!(executed_f64(&task), None);
}

#[test]
fn the_threshold_declines_small_tasks() {
    let left = varied(2, 2, 1);
    let right = varied(2, 2, 2);
    let task = GemmTask::new(&left, [2, 1], &right, [2, 1], 2, 2, 2);
    let left32: Vec<f32> = left.iter().map(|&value| value as f32).collect();
    let right32: Vec<f32> = right.iter().map(|&value| value as f32).collect();
    let task32 = GemmTask::new(&left32, [2, 1], &right32, [2, 1], 2, 2, 2);
    assert_eq!(super::gemm_f64(&task), None);
    assert_eq!(gemm_f32(&task32), None);
}

#[test]
fn repeated_products_answer_bitwise_identically() {
    // Accelerate parallelizes internally; per the house rule its
    // run-to-run determinism is verified, not assumed. The observed
    // answer is recorded in notes/accelerate-backend.md.
    let size = 1024;
    let left = varied(size, size, 5);
    let right = varied(size, size, 6);
    let task = GemmTask::new(&left, [size, 1], &right, [size, 1], size, size, size);
    let first = executed_f64(&task).expect("a plain square product");
    let second = executed_f64(&task).expect("a plain square product");
    let first_bits: Vec<u64> = first.iter().map(|value| value.to_bits()).collect();
    let second_bits: Vec<u64> = second.iter().map(|value| value.to_bits()).collect();
    assert_eq!(first_bits, second_bits);
}

/// One accuracy case of the vForce grid: the operation, its valid
/// inputs, and libm's scalar answer.
type MapCaseF64<'inputs> = (MapOperation, &'inputs [f64], fn(f64) -> f64);
type MapCaseF32<'inputs> = (MapOperation, &'inputs [f32], fn(f32) -> f32);

#[test]
fn vforce_maps_match_libm_within_ulps() {
    let inputs_f64: Vec<f64> = (0..4096)
        .map(|index| ((index * 7) % 23 - 11) as f64 / 4.0)
        .collect();
    let positives_f64: Vec<f64> = inputs_f64.iter().map(|value| value.abs() + 0.25).collect();
    let inputs_f32: Vec<f32> = inputs_f64.iter().map(|&value| value as f32).collect();
    let positives_f32: Vec<f32> = positives_f64.iter().map(|&value| value as f32).collect();

    let cases_f64: [MapCaseF64<'_>; 4] = [
        (MapOperation::Exp, &inputs_f64, f64::exp),
        (MapOperation::Ln, &positives_f64, f64::ln),
        (MapOperation::Sqrt, &positives_f64, f64::sqrt),
        (MapOperation::Tanh, &inputs_f64, f64::tanh),
    ];
    for (operation, elements, scalar) in cases_f64 {
        let mapped = map_f64(operation, elements).expect("above the threshold");
        for (actual, element) in mapped.iter().zip(elements) {
            let expected = scalar(*element);
            assert!(
                (actual - expected).abs() <= 4.0 * f64::EPSILON * (1.0 + expected.abs()),
                "{operation:?}({element}) = {actual}, libm answers {expected}"
            );
        }
    }
    let cases_f32: [MapCaseF32<'_>; 4] = [
        (MapOperation::Exp, &inputs_f32, f32::exp),
        (MapOperation::Ln, &positives_f32, f32::ln),
        (MapOperation::Sqrt, &positives_f32, f32::sqrt),
        (MapOperation::Tanh, &inputs_f32, f32::tanh),
    ];
    for (operation, elements, scalar) in cases_f32 {
        let mapped = map_f32(operation, elements).expect("above the threshold");
        for (actual, element) in mapped.iter().zip(elements) {
            let expected = scalar(*element);
            assert!(
                (actual - expected).abs() <= 4.0 * f32::EPSILON * (1.0 + expected.abs()),
                "{operation:?}({element}) = {actual}, libm answers {expected}"
            );
        }
    }
}

#[test]
fn small_maps_decline_to_the_scalar_path() {
    let elements = [1.0_f32; 16];
    assert_eq!(map_f32(MapOperation::Tanh, &elements), None);
    let elements = [1.0_f64; 16];
    assert_eq!(map_f64(MapOperation::Exp, &elements), None);
}

#[test]
fn tensor_maps_route_contiguous_buffers_and_fall_back_on_views() {
    let elements: Vec<f32> = (0..2048)
        .map(|index| (index % 37) as f32 / 9.0 - 2.0)
        .collect();
    let tensor = Tensor::new([32, 64], elements.clone());
    let mapped = tensor.tanh().to_vec();
    for (actual, element) in mapped.iter().zip(&elements) {
        let expected = element.tanh();
        assert!((actual - expected).abs() <= 4.0 * f32::EPSILON * (1.0 + expected.abs()));
    }

    // A transposed view is not contiguous, so it takes the scalar
    // path: bitwise-identical to libm, proving the fallback.
    let transposed = tensor.transpose();
    let scalar: Vec<u32> = transposed
        .to_vec()
        .iter()
        .map(|element| element.tanh().to_bits())
        .collect();
    let through_map: Vec<u32> = transposed
        .tanh()
        .to_vec()
        .iter()
        .map(|element| element.to_bits())
        .collect();
    assert_eq!(scalar, through_map);
}

#[test]
fn training_runs_through_the_backend_end_to_end() {
    // A batch-64 regression whose products sit above the threshold,
    // so forward and backward (transposed views included) route
    // through cblas against the real tape.
    let tape = Tape::new();
    let inputs = tape.input(Tensor::filled([64, 8], 0.5_f64));
    let targets = tape.input(Tensor::filled([64, 1], 1.0_f64));
    let mut initializer = init::uniform(11, 0.3);
    let weights = tape.parameter(initializer(&Shape::new([8, 64])));
    let output_weights = tape.parameter(initializer(&Shape::new([64, 1])));
    let hidden = inputs.matmul(weights).tanh();
    let prediction = hidden.matmul(output_weights);
    let error = prediction - targets;
    let loss = (error * error).sum();
    let loss_symbol = loss.symbol();

    let network = tape.into_network();
    let mut parameters = network.parameters();
    let mut last_loss = f64::INFINITY;
    for _ in 0..50 {
        let run = network.forward(&parameters, []);
        last_loss = run.of(loss_symbol).to_vec()[0];
        let gradients = run.backward(loss_symbol);
        parameters = parameters.step(&gradients, |weight, gradient| {
            weight.clone() - Tensor::filled(gradient.shape(), 0.001) * gradient.clone()
        });
    }
    assert!(
        last_loss < 1.0,
        "training through the backend did not converge: {last_loss}"
    );
}

#[test]
fn batch_norm_matches_the_hand_rolled_reference() {
    let (batch, features) = (64, 64);
    let input: Vec<f64> = (0..batch * features)
        .map(|index| ((index * 7 % 23) as f64 - 11.0) / 4.0)
        .collect();
    let scale: Vec<f64> = (0..features).map(|j| 0.5 + (j as f64) / 100.0).collect();
    let shift: Vec<f64> = (0..features).map(|j| (j as f64) / 50.0 - 0.5).collect();
    let epsilon = 1.0e-5;
    let task = BatchNormTask::new(&input, &scale, &shift, epsilon, batch, features);
    let normalized = batch_norm_f64(&task).expect("the task clears the threshold");
    for feature in 0..features {
        let mut mean = 0.0;
        for row in 0..batch {
            mean += input[row * features + feature];
        }
        mean /= batch as f64;
        let mut variance = 0.0;
        for row in 0..batch {
            let centered = input[row * features + feature] - mean;
            variance += centered * centered;
        }
        variance /= batch as f64;
        let close = |actual: f64, expected: f64| {
            (actual - expected).abs() <= 1.0e-9 * (1.0 + expected.abs())
        };
        assert!(close(normalized.mean[feature], mean));
        assert!(close(normalized.variance[feature], variance));
        for row in 0..batch {
            let expected = (input[row * features + feature] - mean) / (variance + epsilon).sqrt()
                * scale[feature]
                + shift[feature];
            assert!(close(normalized.output[row * features + feature], expected));
        }
    }
}

#[test]
fn batch_norm_declines_below_the_threshold() {
    let input32 = vec![0.5_f32; 4];
    let ones32 = vec![1.0_f32; 2];
    let task32 = BatchNormTask::new(&input32, &ones32, &ones32, 1.0e-5_f32, 2, 2);
    assert!(batch_norm_f32(&task32).is_none());
    let input64 = vec![0.5_f64; 4];
    let ones64 = vec![1.0_f64; 2];
    let task64 = BatchNormTask::new(&input64, &ones64, &ones64, 1.0e-5_f64, 2, 2);
    assert!(batch_norm_f64(&task64).is_none());
}
