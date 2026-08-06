//! Benchmarks of elementwise operations over broadcast views.
//!
//! Cases mirror the expansions `broadcast_to` records: a bias-style
//! row spread along the leading axis, a column spread along the
//! trailing axis, an outer-product pair of spreads, and a
//! transcendental map over a spread view. Throughput is in logical
//! elements per second, so a path that touches only the distinct
//! elements of a view shows up as higher throughput, not less work.

use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use poorgrad::{Elementary, Shape, Tensor, Tensorial};

/// Builds a `[len]` tensor with a deterministic, cheap fill.
fn filled(len: usize, seed: u64) -> Tensor<f32> {
    let elements: Vec<f32> = (0..len)
        .map(|index| {
            ((index as u64).wrapping_mul(2654435761).wrapping_add(seed) % 1000) as f32 / 500.0 - 1.0
        })
        .collect();
    Tensor::new([len], elements)
}

fn broadcast(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("broadcast");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(2));

    let rows = 2048;
    let columns = 1024;
    let volume = rows * columns;
    let matrix = filled(volume, 1).reshape(Shape::new([rows, columns]));
    let row = filled(columns, 2);
    let column = filled(rows, 3);
    let row_spread = row.broadcast_along(0, &matrix);
    let column_spread = column.broadcast_along(1, &matrix);

    group.throughput(Throughput::Elements(volume as u64));
    group.bench_function("f32/add-row-spread-2m", |bencher| {
        bencher.iter(|| matrix.clone() + row_spread.clone());
    });
    group.throughput(Throughput::Elements(volume as u64));
    group.bench_function("f32/add-column-spread-2m", |bencher| {
        bencher.iter(|| matrix.clone() + column_spread.clone());
    });
    group.throughput(Throughput::Elements(volume as u64));
    group.bench_function("f32/multiply-outer-spreads-2m", |bencher| {
        bencher.iter(|| column_spread.clone() * row_spread.clone());
    });
    group.throughput(Throughput::Elements(volume as u64));
    group.bench_function("f32/exp-row-spread-2m", |bencher| {
        bencher.iter(|| row_spread.exp());
    });

    group.finish();
}

criterion_group!(benches, broadcast);
criterion_main!(benches);
