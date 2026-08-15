//! Grades the three training routes of the `cifar10` convnet against
//! each other — the CIFAR memory story the `differentiate` design
//! left open: do conv gradients recorded through `Fold` beat the
//! engine backward's retain-all and compact-remat postures at scale?
//!
//! The model, seeds, and batch schedule are identical to `cifar10`,
//! so all three routes train bit-identically; what changes is where
//! the gradients come from and what the plan retains:
//!
//! - `engine`: `compile_training(.., Retention::All)` + `backward`.
//! - `compact`: `compile_training(.., Retention::Compact)` +
//!   `backward`, rematerializing the dropped intermediates.
//! - `recorded`: `differentiate` + one forward-only plan over
//!   `[loss, gradients...]` + `recorded_gradients` — no backward
//!   pass executes at all.
//!
//! One route runs per process so an external monitor attributes the
//! peak RSS cleanly; pick it with `POORGRAD_ROUTE` and the step
//! count with `POORGRAD_STEPS`. Each step prints nothing; the end
//! prints the first and final losses as exact bit patterns, so route
//! parity is checked by diffing two lines.
//!
//! Run with: `/usr/bin/time -l cargo run --release --example
//! cifar10_grading` (set `POORGRAD_ROUTE=engine|compact|recorded`).

mod dataset;

use std::time::Instant;

use poorgrad::{
    Conv2d, Linear, Module, Network, Retention, Shape, Symbol, Tensor, Tensorial, Value,
    cross_entropy, init, max_pool,
};

use dataset::{Split, load, shuffle};

/// The image side length; CIFAR-10 images are `32 x 32`.
const IMAGE_SIDE: usize = 32;

/// How many values one image holds: three channel planes.
const PIXELS: usize = 3 * IMAGE_SIDE * IMAGE_SIDE;

/// How many classes the head scores.
const CLASSES: usize = 10;

/// How many samples each training step feeds.
const BATCH_LEN: usize = 64;

/// How many filters the three convolution stages learn.
const FILTERS: [usize; 3] = [16, 32, 64];

/// The flattened feature length after three 2x2 pools: `64 * 4 * 4`.
const FLAT_LEN: usize = FILTERS[2] * (IMAGE_SIDE / 8) * (IMAGE_SIDE / 8);

/// The model's layers, holding parameter symbols across generations;
/// identical to `cifar10`, including every seed.
struct Model {
    conv_1: Conv2d<Tensor<f32>>,
    conv_2: Conv2d<Tensor<f32>>,
    conv_3: Conv2d<Tensor<f32>>,
    head: Linear<Tensor<f32>>,
}

impl Model {
    /// Allocates the parameters on `network` exactly as `cifar10`
    /// does, so the routes train the same trajectory it would.
    fn new(network: &Network<Tensor<f32>>) -> Self {
        let conv_1_weights =
            init::normal(21, (2.0 / 27.0_f64).sqrt())(&Shape::new([FILTERS[0], 3, 3, 3]));
        let conv_2_weights =
            init::normal(22, (2.0 / 144.0_f64).sqrt())(&Shape::new([FILTERS[1], FILTERS[0], 3, 3]));
        let conv_3_weights =
            init::normal(23, (2.0 / 288.0_f64).sqrt())(&Shape::new([FILTERS[2], FILTERS[1], 3, 3]));
        let mut head_weights = init::kaiming(24);
        Self {
            conv_1: Conv2d::new(
                network,
                conv_1_weights,
                Tensor::filled([FILTERS[0]], 0.0),
                1,
                1,
            ),
            conv_2: Conv2d::new(
                network,
                conv_2_weights,
                Tensor::filled([FILTERS[1]], 0.0),
                1,
                1,
            ),
            conv_3: Conv2d::new(
                network,
                conv_3_weights,
                Tensor::filled([FILTERS[2]], 0.0),
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
    /// `[rows, 10]` logits: conv, rectify, pool, three times, then
    /// flatten and score.
    fn express<'network>(
        &self,
        network: &'network Network<Tensor<f32>>,
        images: Value<'network, Tensor<f32>>,
        rows: usize,
    ) -> Value<'network, Tensor<f32>> {
        let stage_1 = max_pool(self.conv_1.express(network, images).relu(), 2, 2);
        let stage_2 = max_pool(self.conv_2.express(network, stage_1).relu(), 2, 2);
        let stage_3 = max_pool(self.conv_3.express(network, stage_2).relu(), 2, 2);
        self.head
            .express(network, stage_3.reshape([rows, FLAT_LEN]))
    }

    /// Returns the parameter symbols in a fixed order, for `wrt` and
    /// for pairing with their recorded gradients.
    fn parameters(&self) -> [Symbol; 8] {
        [
            self.conv_1.weights(),
            self.conv_1.bias(),
            self.conv_2.weights(),
            self.conv_2.bias(),
            self.conv_3.weights(),
            self.conv_3.bias(),
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
        Tensor::new([indices.len(), 3, IMAGE_SIDE, IMAGE_SIDE], pixels),
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
        [BATCH_LEN, 3, IMAGE_SIDE, IMAGE_SIDE],
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
            network.compile_training(loss_symbol, [], Retention::All),
            Vec::new(),
        ),
        "compact" => (
            network.compile_training(loss_symbol, [], Retention::Compact),
            Vec::new(),
        ),
        "recorded" => {
            let gradient_symbols = network.differentiate(loss_symbol, parameter_symbols);
            println!(
                "recorded the chain rule: {forward_nodes} forward nodes + {} gradient nodes",
                network.len() - forward_nodes
            );
            (
                network.compile(
                    std::iter::once(loss_symbol).chain(gradient_symbols.iter().copied()),
                    [],
                ),
                gradient_symbols,
            )
        }
        other => panic!("unknown POORGRAD_ROUTE {other:?}; use engine, compact, or recorded"),
    };
    for line in plan
        .describe()
        .lines()
        .filter(|line| line.starts_with("plan:") || line.starts_with("live volume:"))
    {
        println!("{line}");
    }

    let mut order: Vec<usize> = (0..train.len()).collect();
    let mut shuffle_state: u64 = 3;
    shuffle(&mut order, &mut shuffle_state);

    let fast = Tensor::new([], [0.05_f32]);
    let slow = Tensor::new([], [0.005_f32]);
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
