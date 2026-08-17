//! Benchmarks of the dense matrix-multiplication path.
//!
//! Cases exercise the payload's `matmul` directly, without a network,
//! so the numbers isolate the kernel: square products at growing
//! sizes for both element types, a transposed right operand (the
//! backward pass's most common view), and the makemore-sized
//! rectangle as the workload reality check. Throughput is reported
//! in elements, one element per floating-point operation (`2 * m *
//! n * k` per product), so Melem/s reads as MFLOP/s.

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use topos::{Tensor, Tensorial};

/// Builds a `[rows, columns]` tensor with a deterministic, cheap fill.
fn filled_f64(rows: usize, columns: usize, seed: u64) -> Tensor<f64> {
    let elements: Vec<f64> = (0..rows * columns)
        .map(|index| {
            ((index as u64).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f64 / 1000.0
        })
        .collect();
    Tensor::new([rows, columns], elements)
}

/// Builds the `f32` twin of [`filled_f64`].
fn filled_f32(rows: usize, columns: usize, seed: u64) -> Tensor<f32> {
    let elements: Vec<f32> = (0..rows * columns)
        .map(|index| {
            ((index as u64).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32 / 1000.0
        })
        .collect();
    Tensor::new([rows, columns], elements)
}

fn gemm(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("gemm");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    for size in [64usize, 256, 512] {
        let flops = (2 * size * size * size) as u64;

        let left = filled_f64(size, size, 1);
        let right = filled_f64(size, size, 2);
        group.throughput(Throughput::Elements(flops));
        group.bench_function(format!("f64/square-{size}"), |bencher| {
            bencher.iter(|| left.matmul(&right));
        });

        let left = filled_f32(size, size, 1);
        let right = filled_f32(size, size, 2);
        group.throughput(Throughput::Elements(flops));
        group.bench_function(format!("f32/square-{size}"), |bencher| {
            bencher.iter(|| left.matmul(&right));
        });
    }

    // The backward pass's shapes: a transposed operand on either side,
    // as `gradient . b^T` and `a^T . gradient` produce them.
    let size = 256usize;
    let flops = (2 * size * size * size) as u64;

    let left = filled_f64(size, size, 1);
    let right_transposed = filled_f64(size, size, 2).transpose();
    group.throughput(Throughput::Elements(flops));
    group.bench_function("f64/transposed-b-256", |bencher| {
        bencher.iter(|| left.matmul(&right_transposed));
    });

    let left_transposed = filled_f64(size, size, 1).transpose();
    let right = filled_f64(size, size, 2);
    group.throughput(Throughput::Elements(flops));
    group.bench_function("f64/transposed-a-256", |bencher| {
        bencher.iter(|| left_transposed.matmul(&right));
    });

    // The makemore hidden layer, batch 64: the honest size of today's
    // examples.
    let contexts = filled_f64(64, 30, 1);
    let weights = filled_f64(30, 100, 2);
    group.throughput(Throughput::Elements((2 * 64 * 30 * 100) as u64));
    group.bench_function("f64/makemore-64x30x100", |bencher| {
        bencher.iter(|| contexts.matmul(&weights));
    });

    // The sizes where a GPU can pay for its dispatch: the `metal`
    // feature's territory, and the accelerate-versus-metal crossover.
    for size in [1024usize, 2048] {
        let flops = (2 * size * size * size) as u64;
        let left = filled_f32(size, size, 1);
        let right = filled_f32(size, size, 2);
        group.throughput(Throughput::Elements(flops));
        group.bench_function(format!("f32/square-{size}"), |bencher| {
            bencher.iter(|| left.matmul(&right));
        });
    }
    let size = 1024usize;
    let left = filled_f64(size, size, 1);
    let right = filled_f64(size, size, 2);
    group.throughput(Throughput::Elements((2 * size * size * size) as u64));
    group.bench_function("f64/square-1024", |bencher| {
        bencher.iter(|| left.matmul(&right));
    });

    group.finish();
}

criterion_group!(benches, gemm);
criterion_main!(benches);
