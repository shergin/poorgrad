use crate::GemmTask;

use super::gemm::{Kernel, executed};
use super::{context, gemm_f32};

/// Computes the product of two row-major matrices on the host as the
/// tolerance reference.
fn reference(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut elements = Vec::with_capacity(m * n);
    for row in 0..m {
        for column in 0..n {
            let mut total = a[row * k] * b[column];
            for step in 1..k {
                total += a[row * k + step] * b[step * n + column];
            }
            elements.push(total);
        }
    }
    elements
}

/// Returns distinct, sign-varied row-major values.
fn varied(rows: usize, columns: usize, seed: i64) -> Vec<f32> {
    (0..(rows * columns) as i64)
        .map(|index| ((index * 7 + seed * 13) % 23 - 11) as f32 / 4.0)
        .collect()
}

/// Asserts elementwise closeness under the k-scaled tolerance; the
/// GPU sums in tile order, so bitwise equality is not the contract.
fn assert_close(actual: &[f32], expected: &[f32], k: usize) {
    let tolerance = 8.0 * f32::EPSILON * (k as f32).sqrt();
    for (actual, expected) in actual.iter().zip(expected) {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "{actual} differs from {expected} beyond tolerance (k = {k})"
        );
    }
}

/// Returns the transpose-stored buffer of a logical matrix, read
/// back through a transposed view's strides.
fn transposed(logical: &[f32], rows: usize, columns: usize) -> Vec<f32> {
    let mut buffer = vec![0.0_f32; rows * columns];
    for row in 0..rows {
        for column in 0..columns {
            buffer[column * rows + row] = logical[row * columns + column];
        }
    }
    buffer
}

#[test]
fn both_kernels_match_the_reference_across_the_grid() {
    let context = context().expect("this machine has a Metal device");
    for (m, k, n) in [
        (1, 1, 1),
        (2, 3, 4),
        (5, 8, 13),
        (16, 16, 16),
        (63, 64, 65),
        (64, 64, 64),
        (100, 127, 3),
        (128, 100, 64),
        (65, 129, 66),
    ] {
        let left = varied(m, k, 1);
        let right = varied(k, n, 2);
        let expected = reference(&left, &right, m, k, n);
        let task = GemmTask::new(&left, [k, 1], &right, [n, 1], m, k, n);
        for kernel in [Kernel::Naive, Kernel::Tiled] {
            let product = executed(context, &task, kernel).expect("the dispatch succeeds");
            assert_close(&product, &expected, k);
        }
    }
}

#[test]
fn transposed_views_match_the_reference() {
    let context = context().expect("this machine has a Metal device");
    let (m, k, n) = (65, 64, 63);
    let left = varied(m, k, 3);
    let right = varied(k, n, 4);
    let expected = reference(&left, &right, m, k, n);

    let left_transposed = transposed(&left, m, k);
    let right_transposed = transposed(&right, k, n);
    for kernel in [Kernel::Naive, Kernel::Tiled] {
        let task = GemmTask::new(&left_transposed, [1, m], &right, [n, 1], m, k, n);
        assert_close(
            &executed(context, &task, kernel).expect("a transposed left operand"),
            &expected,
            k,
        );
        let task = GemmTask::new(&left, [k, 1], &right_transposed, [1, k], m, k, n);
        assert_close(
            &executed(context, &task, kernel).expect("a transposed right operand"),
            &expected,
            k,
        );
    }
}

#[test]
fn small_and_gemv_tasks_decline() {
    let left = varied(64, 64, 1);
    let right = varied(64, 64, 2);
    let task = GemmTask::new(&left, [64, 1], &right, [64, 1], 64, 64, 64);
    assert_eq!(gemm_f32(&task), None);

    let row = varied(1, 1024, 3);
    let wide = varied(1024, 1024, 4);
    let task = GemmTask::new(&row, [1024, 1], &wide, [1024, 1], 1, 1024, 1024);
    assert_eq!(gemm_f32(&task), None);
}

// With `accelerate` also compiled, the chain's first arm answers
// instead, so the routing assertion holds only for metal-only builds.
#[cfg(not(feature = "accelerate"))]
#[test]
fn the_chain_routes_large_products_here() {
    // Above the threshold the chain's first arm answers; compare its
    // product against this module's own dispatch to prove the wiring
    // (the tile order is fixed, so the two runs agree bitwise).
    let size = 1024;
    let left = varied(size, size, 5);
    let right = varied(size, size, 6);
    let task = GemmTask::new(&left, [size, 1], &right, [size, 1], size, size, size);
    let through_chain = crate::backend::gemm_f32(&task).expect("the chain accepts a 1024-cube");
    let direct = executed(
        context().expect("this machine has a Metal device"),
        &task,
        Kernel::Tiled,
    )
    .expect("the dispatch succeeds");
    let chain_bits: Vec<u32> = through_chain.iter().map(|value| value.to_bits()).collect();
    let direct_bits: Vec<u32> = direct.iter().map(|value| value.to_bits()).collect();
    assert_eq!(chain_bits, direct_bits);
}
