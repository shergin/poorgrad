//! Trains a small character-level transformer — the attention act of
//! the makemore series, and the consumer that closes the transformer
//! rung: every piece is composition over the existing op set.
//!
//! The batch packs its samples into one `[BATCH_LEN * CONTEXT_LEN]`
//! token row, so each head's attention is a single rank-2 matmul pair
//! over the packed axis; a block-diagonal causal mask (an additive
//! `0 / -inf` leaf) keeps samples independent and time causal, which
//! is the sequence-packing idiom and the reason no batched matmul is
//! needed. Masked softmax records as the mask added before the axis
//! softmax, heads record as a loop of rank-2 attentions joined by
//! `concat`, and the per-sample prediction rows come back through a
//! one-hot `gather`. The block is pre-norm: attention and feed-forward
//! each read an `RmsNorm` of their input and add onto the residual
//! stream. Loss is measured at the last context position only, so the
//! number is comparable to the MLP acts (uniform cost: ln 27 ~ 3.30).
//!
//! Run with: `cargo run --release --example makemore_transformer`

mod chart;
mod corpus;

use std::time::Instant;

use poorgrad::{
    Compile, Memory, Network, RmsNorm, Shape, Tensor, Tensorial, Value, concat, cross_entropy, init,
};

use chart::loss_chart;
use corpus::{VOCABULARY_LEN, draw, from_token, load_names, shuffle, training_samples};

/// How many characters of history the model attends over before
/// predicting the next one.
const CONTEXT_LEN: usize = 8;

/// How many dimensions the residual stream has.
const EMBED_DIM: usize = 32;

/// How many attention heads split the stream.
const HEAD_COUNT: usize = 2;

/// How many dimensions each head reads and writes.
const HEAD_DIM: usize = EMBED_DIM / HEAD_COUNT;

/// How many neurons the feed-forward hidden layer has.
const HIDDEN_LEN: usize = 4 * EMBED_DIM;

/// How many samples each training step packs into the attention row.
const BATCH_LEN: usize = 32;

/// How many token rows the packed training batch holds.
const PACKED_LEN: usize = BATCH_LEN * CONTEXT_LEN;

/// One attention head's projections, each `[EMBED_DIM, HEAD_DIM]`.
struct Head<'network> {
    query: Value<'network, Tensor<f32>>,
    key: Value<'network, Tensor<f32>>,
    value: Value<'network, Tensor<f32>>,
}

/// The model's parameters as recorded proxies: embeddings, one
/// pre-norm attention block, and the language-model head.
struct Model<'network> {
    embeddings: Value<'network, Tensor<f32>>,
    positions: Value<'network, Tensor<f32>>,
    heads: Vec<Head<'network>>,
    projection: Value<'network, Tensor<f32>>,
    attention_norm: RmsNorm<Tensor<f32>>,
    hidden_weights: Value<'network, Tensor<f32>>,
    output_weights: Value<'network, Tensor<f32>>,
    hidden_norm: RmsNorm<Tensor<f32>>,
    final_norm: RmsNorm<Tensor<f32>>,
    logit_weights: Value<'network, Tensor<f32>>,
    logit_bias: Value<'network, Tensor<f32>>,
    scale: Value<'network, Tensor<f32>>,
}

impl<'network> Model<'network> {
    /// Allocates the parameters on `network`: embedding tables for
    /// characters and positions, the heads' projections, the block's
    /// two norms and feed-forward, the final norm, and the affine
    /// logit head. The attention scale `1 / sqrt(HEAD_DIM)` rides
    /// along as a single-value leaf.
    fn new(network: &'network Network<Tensor<f32>>) -> Self {
        let mut weights = init::xavier(7);
        let ones = Tensor::filled([EMBED_DIM], 1.0);
        let epsilon = Tensor::filled([], 1e-5);
        Self {
            embeddings: network.parameter(init::normal(8, 1.0)(&Shape::new([
                VOCABULARY_LEN,
                EMBED_DIM,
            ]))),
            positions: network
                .parameter(init::normal(9, 1.0)(&Shape::new([CONTEXT_LEN, EMBED_DIM]))),
            heads: (0..HEAD_COUNT)
                .map(|_| Head {
                    query: network.parameter(weights(&Shape::new([EMBED_DIM, HEAD_DIM]))),
                    key: network.parameter(weights(&Shape::new([EMBED_DIM, HEAD_DIM]))),
                    value: network.parameter(weights(&Shape::new([EMBED_DIM, HEAD_DIM]))),
                })
                .collect(),
            projection: network.parameter(weights(&Shape::new([EMBED_DIM, EMBED_DIM]))),
            attention_norm: RmsNorm::new(network, ones.clone(), epsilon.clone()),
            hidden_weights: network.parameter(weights(&Shape::new([EMBED_DIM, HIDDEN_LEN]))),
            output_weights: network.parameter(weights(&Shape::new([HIDDEN_LEN, EMBED_DIM]))),
            hidden_norm: RmsNorm::new(network, ones.clone(), epsilon.clone()),
            final_norm: RmsNorm::new(network, ones, epsilon),
            logit_weights: network.parameter(weights(&Shape::new([EMBED_DIM, VOCABULARY_LEN]))),
            logit_bias: network.parameter(weights(&Shape::new([VOCABULARY_LEN]))),
            scale: network.leaf(Tensor::filled([], 1.0 / (HEAD_DIM as f32).sqrt())),
        }
    }

    /// Records the block over one packed token row and returns the
    /// normalized residual stream: embeddings in, attention and
    /// feed-forward added on, final norm out.
    ///
    /// `tokens` and `positions` are one-hot selections over the packed
    /// row; `mask` is the additive `0 / -inf` block-causal leaf shaped
    /// `[rows, rows]`.
    fn states(
        &self,
        network: &'network Network<Tensor<f32>>,
        tokens: Value<'network, Tensor<f32>>,
        positions: Value<'network, Tensor<f32>>,
        mask: Value<'network, Tensor<f32>>,
    ) -> Value<'network, Tensor<f32>> {
        let stream = self.embeddings.gather(tokens) + self.positions.gather(positions);

        // Pre-norm attention: every head attends over the same
        // normalized stream, and `concat` joins the head outputs along
        // the feature axis.
        let normalized = self.attention_norm.express(network, stream);
        let heads: Vec<Value<'network, Tensor<f32>>> = self
            .heads
            .iter()
            .map(|head| {
                let query = normalized.matmul(head.query);
                let key = normalized.matmul(head.key);
                let scores = query.matmul(key.transpose());
                let scaled = scores * self.scale.broadcast_like(scores);
                let weights = (scaled + mask).softmax(1);
                weights.matmul(normalized.matmul(head.value))
            })
            .collect();
        let stream = stream + concat(&heads, 1).matmul(self.projection);

        // Pre-norm feed-forward onto the residual stream.
        let normalized = self.hidden_norm.express(network, stream);
        let stream = stream
            + normalized
                .matmul(self.hidden_weights)
                .relu()
                .matmul(self.output_weights);

        self.final_norm.express(network, stream)
    }

    /// Records the logits of the rows picked by the one-hot
    /// `extraction` — the last context position of each packed sample.
    fn logits(
        &self,
        states: Value<'network, Tensor<f32>>,
        extraction: Value<'network, Tensor<f32>>,
    ) -> Value<'network, Tensor<f32>> {
        let product = states.gather(extraction).matmul(self.logit_weights);
        product + self.logit_bias.broadcast_along(0, product)
    }
}

/// Returns the additive attention mask for `samples` packed windows of
/// `CONTEXT_LEN` tokens: zero where the key's row is in the same
/// sample and not after the query's row, negative infinity elsewhere.
fn block_causal_mask(samples: usize) -> Tensor<f32> {
    let rows = samples * CONTEXT_LEN;
    let mut elements = Vec::with_capacity(rows * rows);
    for query in 0..rows {
        for key in 0..rows {
            let same_sample = query / CONTEXT_LEN == key / CONTEXT_LEN;
            let causal = key % CONTEXT_LEN <= query % CONTEXT_LEN;
            elements.push(if same_sample && causal {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    Tensor::new([rows, rows], elements)
}

fn main() {
    let names = load_names();
    let mut samples = training_samples::<CONTEXT_LEN>(&names);
    let mut shuffle_state: u64 = 9;
    shuffle(&mut samples, &mut shuffle_state);
    println!("loaded {} names, {} samples", names.len(), samples.len());

    let network = Network::new();
    let model = Model::new(&network);

    // The training expression over the packed batch: per-sample
    // prediction rows come back through the fixed one-hot extraction.
    let tokens = network.input(Tensor::selection(vec![0; PACKED_LEN], VOCABULARY_LEN, 1.0));
    let positions = network.leaf(Tensor::selection(
        (0..PACKED_LEN)
            .map(|row| row % CONTEXT_LEN)
            .collect::<Vec<_>>(),
        CONTEXT_LEN,
        1.0,
    ));
    let mask = network.leaf(block_causal_mask(BATCH_LEN));
    let extraction = network.leaf(Tensor::selection(
        (0..BATCH_LEN)
            .map(|sample| sample * CONTEXT_LEN + CONTEXT_LEN - 1)
            .collect::<Vec<_>>(),
        PACKED_LEN,
        1.0,
    ));
    let targets = network.input(Tensor::selection(vec![0; BATCH_LEN], VOCABULARY_LEN, 1.0));
    let states = model.states(&network, tokens, positions, mask);
    let loss = cross_entropy(model.logits(states, extraction), targets);

    // The sampling twin is the same expression over one window.
    let sample_tokens = network.input(Tensor::selection(vec![0; CONTEXT_LEN], VOCABULARY_LEN, 1.0));
    let sample_positions = network.leaf(Tensor::selection(
        (0..CONTEXT_LEN).collect::<Vec<_>>(),
        CONTEXT_LEN,
        1.0,
    ));
    let sample_mask = network.leaf(block_causal_mask(1));
    let sample_extraction =
        network.leaf(Tensor::selection(vec![CONTEXT_LEN - 1], CONTEXT_LEN, 1.0));
    let sample_states = model.states(&network, sample_tokens, sample_positions, sample_mask);
    let sample_probabilities = model.logits(sample_states, sample_extraction).softmax(1);

    let tokens_symbol = tokens.symbol();
    let targets_symbol = targets.symbol();
    let loss_symbol = loss.symbol();
    let sample_tokens_symbol = sample_tokens.symbol();
    let sample_probabilities_symbol = sample_probabilities.symbol();
    let recorded_nodes = network.len();

    // Compile once: training keeps only the loss, sampling is
    // forward-only.
    let training_plan =
        network.compile(Compile::roots([loss_symbol]).engine_backward(Memory::Retain));
    let sampling_plan = network.compile(Compile::roots([sample_probabilities_symbol]));

    let fast = Tensor::new([], [0.1]);
    let slow = Tensor::new([], [0.01]);
    let mut network = network;
    let mut window_loss = 0.0;
    let mut losses = Vec::new();
    let training = Instant::now();
    for step in 0..5000 {
        let start = (step * BATCH_LEN) % (samples.len() - BATCH_LEN);
        let batch = &samples[start..start + BATCH_LEN];
        let batch_tokens: Vec<usize> = batch
            .iter()
            .flat_map(|(context, _)| context.iter().copied())
            .collect();
        let batch_targets: Vec<usize> = batch.iter().map(|&(_, next)| next).collect();

        let loss_value = network.resolve(loss_symbol);
        let run = training_plan.forward(
            &network,
            [
                (
                    tokens_symbol,
                    Tensor::selection(batch_tokens, VOCABULARY_LEN, 1.0),
                ),
                (
                    targets_symbol,
                    Tensor::selection(batch_targets, VOCABULARY_LEN, 1.0),
                ),
            ],
        );
        let batch_loss = run.of(loss_value).to_vec()[0];
        losses.push(batch_loss);
        if step == 0 {
            println!(
                "step 0: minibatch loss = {batch_loss:.4} (a uniform model costs ln 27 ~ 3.30)"
            );
        }
        window_loss += batch_loss;
        if (step + 1) % 500 == 0 {
            println!(
                "steps {:4}..{:4}: mean minibatch loss = {:.4}",
                step + 1 - 500,
                step + 1,
                window_loss / 500.0
            );
            window_loss = 0.0;
        }

        let gradients = run.backward(loss_value);
        let learning_rate = if step < 4000 { &fast } else { &slow };
        network = network.update(&gradients, |parameter, gradient| {
            parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }

    println!(
        "trained {} steps in {:.3}s",
        losses.len(),
        training.elapsed().as_secs_f64()
    );

    assert_eq!(network.len(), recorded_nodes);
    println!("the tape held {recorded_nodes} nodes through every step");
    println!("{}", loss_chart("transformer training", &losses));

    println!("sampled names:");
    let mut state: u64 = 7;
    for _ in 0..10 {
        let mut window = [0usize; CONTEXT_LEN];
        let mut name = String::new();
        loop {
            let run = sampling_plan.forward(
                &network,
                [(
                    sample_tokens_symbol,
                    Tensor::selection(window.to_vec(), VOCABULARY_LEN, 1.0),
                )],
            );
            let row = run
                .of(network.resolve(sample_probabilities_symbol))
                .to_vec();
            let token = draw(&row, &mut state);
            if token == 0 {
                break;
            }
            name.push(from_token(token));
            window.rotate_left(1);
            window[CONTEXT_LEN - 1] = token;
        }
        println!("  {name}");
    }
}
