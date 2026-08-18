//! Benchmarks of graph recording: the per-operation cost of the tape's
//! single mutex, sequentially and under thread contention.

use std::hint::black_box;
use std::thread;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};

use topos::Tape;

const NODES: usize = 10_000;
const THREADS: usize = 4;

fn record(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("record");
    group.sample_size(30);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(NODES as u64));

    group.bench_function("sequential", |bencher| {
        bencher.iter(|| {
            let tape = Tape::new();
            let mut value = tape.leaf(1.0_f64);
            for _ in 0..NODES - 1 {
                value = value * value;
            }
            black_box(tape.len())
        });
    });

    // The same total node count recorded by four threads sharing one
    // tape: measures contention on the tape mutex.
    group.bench_function("concurrent-x4", |bencher| {
        bencher.iter(|| {
            let tape = Tape::new();
            thread::scope(|scope| {
                for _ in 0..THREADS {
                    scope.spawn(|| {
                        let mut value = tape.leaf(1.0_f64);
                        for _ in 0..NODES / THREADS - 1 {
                            value = value * value;
                        }
                    });
                }
            });
            black_box(tape.len())
        });
    });

    group.finish();
}

criterion_group!(benches, record);
criterion_main!(benches);
