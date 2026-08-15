//! Grades the training routes of the `mnist` convnet against each
//! other — the companion measurement to `cifar10_grading`, run on
//! the one consumer where rematerialization ever won. Its 2026-08-14
//! verdict retired remat: the recorded route beat retain-all (~78 vs
//! ~89 ms/step, ~270 vs ~365 MiB peak RSS) and beat remat's own
//! numbers (~98 ms/step, ~330 MiB) on both axes at once.
//!
//! The model, seeds, and batch schedule are identical to `mnist`, so
//! the routes train bit-identically; what changes is where the
//! gradients come from and what the plan retains:
//!
//! - `engine`: `engine_backward()` + `backward` (retain-all).
//! - `recorded`: `differentiate` + one forward-only plan over
//!   `[loss, gradients...]` + `recorded_gradients` — no backward
//!   pass executes at all.
//!
//! One route runs per process so an external monitor attributes the
//! peak RSS cleanly; pick it with `POORGRAD_ROUTE` and the step
//! count with `POORGRAD_STEPS`. The end prints the first and final
//! losses as exact bit patterns, so route parity is checked by
//! diffing two lines.
//!
//! Run with: `/usr/bin/time -l cargo run --release --example
//! mnist_grading` (set `POORGRAD_ROUTE=engine|recorded`).

mod dataset;

use std::time::Instant;

use poorgrad::{
    Compile, Conv2d, Linear, Module, Network, Shape, Symbol, Tensor, Tensorial, Value,
    cross_entropy, init, max_pool,
};

use dataset::{Split, load, shuffle};

/// The image side length; MNIST digits are `28 x 28`.
const IMAGE_SIDE: usize = 28;

/// How many pixels one image holds.
const PIXELS: usize = IMAGE_SIDE * IMAGE_SIDE;

/// How many digit classes the head scores.
const CLASSES: usize = 10;

/// How many samples each training step feeds.
const BATCH_LEN: usize = 64;

/// How many filters the first convolution stage learns.
const FILTERS_1: usize = 8;

/// How many filters the second convolution stage learns.
const FILTERS_2: usize = 16;

/// The flattened feature length after two 2x2 pools: `16 * 7 * 7`.
const FLAT_LEN: usize = FILTERS_2 * (IMAGE_SIDE / 4) * (IMAGE_SIDE / 4);

/// The model's layers, holding parameter symbols across generations;
/// identical to `mnist`, including every seed.
struct Model {
    conv_1: Conv2d<Tensor<f32>>,
    conv_2: Conv2d<Tensor<f32>>,
    head: Linear<Tensor<f32>>,
}

impl Model {
    /// Allocates the parameters on `network` exactly as `mnist` does,
    /// so the routes train the same trajectory it would.
    fn new(network: &Network<Tensor<f32>>) -> Self {
        let conv_1_weights =
            init::normal(11, (2.0 / 9.0_f64).sqrt())(&Shape::new([FILTERS_1, 1, 3, 3]));
        let conv_2_weights =
            init::normal(12, (2.0 / 72.0_f64).sqrt())(&Shape::new([FILTERS_2, FILTERS_1, 3, 3]));
        let mut head_weights = init::kaiming(13);
        Self {
            conv_1: Conv2d::new(
                network,
                conv_1_weights,
                Tensor::filled([FILTERS_1], 0.0),
                1,
                1,
            ),
            conv_2: Conv2d::new(
                network,
                conv_2_weights,
                Tensor::filled([FILTERS_2], 0.0),
                1,
                1,
            ),
            head: Linear::new(
                network,
                head_weights(&Shape::new([FLAT_LEN, CLASSES])),
                head_weights(&Shape::new([CLASSES])),
            ),
        }
    }

    /// Records the model's expression over `images` and returns the
    /// `[rows, 10]` logits: conv, rectify, pool, twice, then flatten
    /// and score.
    fn express<'network>(
        &self,
        network: &'network Network<Tensor<f32>>,
        images: Value<'network, Tensor<f32>>,
        rows: usize,
    ) -> Value<'network, Tensor<f32>> {
        let stage_1 = max_pool(self.conv_1.express(network, images).relu(), 2, 2);
        let stage_2 = max_pool(self.conv_2.express(network, stage_1).relu(), 2, 2);
        self.head
            .express(network, stage_2.reshape([rows, FLAT_LEN]))
    }

    /// Returns the parameter symbols in a fixed order, for `wrt` and
    /// for pairing with their recorded gradients.
    fn parameters(&self) -> [Symbol; 6] {
        [
            self.conv_1.weights(),
            self.conv_1.bias(),
            self.conv_2.weights(),
            self.conv_2.bias(),
            self.head.weights(),
            self.head.bias(),
        ]
    }
}

/// Builds the image and one-hot label payloads for the sample `indices`.
fn batch_payloads(split: &Split, indices: &[usize]) -> (Tensor<f32>, Tensor<f32>) {
    let mut pixels = Vec::with_capacity(indices.len() * PIXELS);
    for &index in indices {
        pixels.extend_from_slice(&split.pixels[index * PIXELS..(index + 1) * PIXELS]);
    }
    let labels: Vec<usize> = indices.iter().map(|&index| split.labels[index]).collect();
    (
        Tensor::new([indices.len(), 1, IMAGE_SIDE, IMAGE_SIDE], pixels),
        Tensor::selection(labels, CLASSES, 1.0),
    )
}

fn main() {
    let route = std::env::var("POORGRAD_ROUTE").unwrap_or_else(|_| "engine".to_string());
    let steps: usize = std::env::var("POORGRAD_STEPS")
        .ok()
        .and_then(|steps| steps.parse().ok())
        .unwrap_or(300);

    let (train, _test) = load();
    println!("route {route}: {steps} steps over {} images", train.len());

    let network = Network::new();
    let model = Model::new(&network);
    let images = network.input(Tensor::filled(
        [BATCH_LEN, 1, IMAGE_SIDE, IMAGE_SIDE],
        0.0_f32,
    ));
    let targets = network.input(Tensor::selection(vec![0; BATCH_LEN], CLASSES, 1.0));
    let loss = cross_entropy(model.express(&network, images, BATCH_LEN), targets);

    let images_symbol = images.symbol();
    let targets_symbol = targets.symbol();
    let loss_symbol = loss.symbol();
    let parameter_symbols = model.parameters();
    let forward_nodes = network.len();

    // The routes differ only here: what the plan computes and where
    // the gradients come from.
    let (plan, gradient_symbols) = match route.as_str() {
        "engine" => (
            network.compile(Compile::roots([loss_symbol]).engine_backward()),
            Vec::new(),
        ),
        "recorded" => {
            let gradient_symbols = network.differentiate(loss_symbol, parameter_symbols);
            println!(
                "recorded the chain rule: {forward_nodes} forward nodes + {} gradient nodes",
                network.len() - forward_nodes
            );
            (
                network.compile(Compile::roots(
                    std::iter::once(loss_symbol).chain(gradient_symbols.iter().copied()),
                )),
                gradient_symbols,
            )
        }
        other => panic!("unknown POORGRAD_ROUTE {other:?}; use engine or recorded"),
    };
    for line in plan
        .describe()
        .lines()
        .filter(|line| line.starts_with("plan:") || line.starts_with("live volume:"))
    {
        println!("{line}");
    }

    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut shuffle_state: u64 = 5;
    shuffle(&mut order, &mut shuffle_state);

    let fast = Tensor::new([], [0.1_f32]);
    let slow = Tensor::new([], [0.01_f32]);
    let mut network = network;
    let mut first_loss = 0.0_f32;
    let mut last_loss = 0.0_f32;
    let training = Instant::now();
    for step in 0..steps {
        let start = (step * BATCH_LEN) % (train.len() - BATCH_LEN);
        let batch = &order[start..start + BATCH_LEN];
        let (batch_images, batch_targets) = batch_payloads(&train, batch);

        let run = plan.forward(
            &network,
            [
                (images_symbol, batch_images),
                (targets_symbol, batch_targets),
            ],
        );
        let batch_loss = run.of(network.resolve(loss_symbol)).to_vec()[0];
        if step == 0 {
            first_loss = batch_loss;
        }
        last_loss = batch_loss;

        let gradients = if route == "recorded" {
            run.recorded_gradients(parameter_symbols.iter().zip(&gradient_symbols).map(
                |(&parameter, &gradient)| (network.resolve(parameter), network.resolve(gradient)),
            ))
        } else {
            run.backward(network.resolve(loss_symbol))
        };
        let learning_rate = if step < steps * 3 / 4 { &fast } else { &slow };
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }
    let elapsed = training.elapsed().as_secs_f64();

    // Exact bit patterns: two routes agree exactly when these two
    // lines match.
    println!(
        "loss step 0: {first_loss:.6} ({:08x}), step {}: {last_loss:.6} ({:08x})",
        first_loss.to_bits(),
        steps - 1,
        last_loss.to_bits(),
    );
    println!(
        "route {route}: {steps} steps in {elapsed:.3}s ({:.1} ms/step)",
        elapsed * 1000.0 / steps as f64
    );
}
