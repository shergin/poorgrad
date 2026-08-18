//! GPT-2 (124M) as a module tree: the released checkpoint's layout,
//! expressed in topos's composition tier.
//!
//! Every struct here is an ordinary [`Module`] implementation — the
//! blocks are structs of [`Linear`]s and [`LayerNorm`]s, the stack of
//! twelve is a [`Sequential`], and attention is a custom module built
//! from two `Linear`s and the public op surface. The tree's `visit`
//! paths mirror the checkpoint's own tensor names (`h.{i}.attn.c_attn`,
//! `ln_f`, ...), so loading the pretrained weights is one
//! [`named_restore`] over the paths the model announces itself; the
//! adapter shrinks to the leaf spellings the module tier and the
//! checkpoint disagree on.
//!
//! The tree is generic over the element type: the same structs record
//! the f32 model and the `Bf16` one, which is the genericity the
//! module design promises.

use topos::checkpoint::named_restore;
use topos::{
    Elementary, LayerNorm, Linear, Module, Parameters, Path, Segment, Sequential, Symbol, Tape,
    Tensor, Value, Visitor, concat, named_parameters,
};

use crate::weights::Weights;

/// How many tokens of context the recorded graph attends over.
pub const CONTEXT_LEN: usize = 256;

/// How many dimensions the residual stream has.
pub const EMBED_DIM: usize = 768;

/// How many attention heads split the stream.
const HEAD_COUNT: usize = 12;

/// How many dimensions each head reads and writes.
const HEAD_DIM: usize = EMBED_DIM / HEAD_COUNT;

/// How many transformer blocks the model stacks.
const LAYER_COUNT: usize = 12;

/// How many tokens the vocabulary holds.
pub const VOCABULARY_LEN: usize = 50257;

/// How many positions the released position table covers.
const POSITION_COUNT: usize = 1024;

/// The GELU tanh approximation the checkpoint was trained with, its
/// constants held as scalar leaves shared by every block — float
/// constants are caller territory, so the facade tier cannot supply
/// this activation.
#[derive(Clone)]
struct Gelu {
    half: Symbol,
    one: Symbol,
    root: Symbol,
    coefficient: Symbol,
}

impl Gelu {
    fn new<E: Elementary + From<f32>>(tape: &Tape<Tensor<E>>) -> Self {
        let scalar = |value: f32| tape.leaf(Tensor::filled([], E::from(value))).symbol();
        Self {
            half: scalar(0.5),
            one: scalar(1.0),
            // The square root of 2 over pi, as the checkpoint's
            // training defined it.
            root: scalar(0.797_884_6),
            coefficient: scalar(0.044_715),
        }
    }
}

impl<E: Elementary> Module<Tensor<E>> for Gelu {
    /// Records `0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3)))`.
    ///
    /// The constants are leaves, not parameters, so the default no-op
    /// `visit` is right: the checkpoint has nothing to restore here.
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        input: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        let half = tape.resolve(self.half);
        let one = tape.resolve(self.one);
        let root = tape.resolve(self.root);
        let coefficient = tape.resolve(self.coefficient);
        let cubic = input * input * input * coefficient.broadcast_like(input);
        let inner = ((input + cubic) * root.broadcast_like(input)).tanh();
        input * (inner + one.broadcast_like(inner)) * half.broadcast_like(input)
    }
}

/// Multi-head causal self-attention over the fused query-key-value
/// projection the checkpoint ships: every head is a rank-2 slice of
/// one `c_attn` output, and the heads concatenate back through
/// `c_proj`.
struct Attention<E> {
    fused: Linear<Tensor<E>>,
    projection: Linear<Tensor<E>>,
    mask: Symbol,
    scale: Symbol,
}

impl<E: Elementary + From<f32>> Attention<E> {
    /// Allocates the projections with placeholder payloads; `mask` and
    /// `scale` are leaves shared by every block.
    fn new(tape: &Tape<Tensor<E>>, mask: Symbol, scale: Symbol) -> Self {
        let zeros = |shape: [usize; 2]| Tensor::filled(shape, E::from(0.0));
        let bias = |extent: usize| Tensor::filled([extent], E::from(0.0));
        Self {
            fused: Linear::new(tape, zeros([EMBED_DIM, 3 * EMBED_DIM]), bias(3 * EMBED_DIM)),
            projection: Linear::new(tape, zeros([EMBED_DIM, EMBED_DIM]), bias(EMBED_DIM)),
            mask,
            scale,
        }
    }
}

impl<E: Elementary> Module<Tensor<E>> for Attention<E> {
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        input: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        let mask = tape.resolve(self.mask);
        let scale = tape.resolve(self.scale);
        let fused = self.fused.express(tape, input);
        let heads: Vec<Value<'tape, Tensor<E>>> = (0..HEAD_COUNT)
            .map(|head| {
                let query = fused.narrow(1, head * HEAD_DIM, HEAD_DIM);
                let key = fused.narrow(1, EMBED_DIM + head * HEAD_DIM, HEAD_DIM);
                let value = fused.narrow(1, 2 * EMBED_DIM + head * HEAD_DIM, HEAD_DIM);
                let scores = query.matmul(key.transpose());
                let weights = (scores * scale.broadcast_like(scores) + mask).softmax(1);
                weights.matmul(value)
            })
            .collect();
        self.projection.express(tape, concat(&heads, 1))
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.enter(Segment::Name("c_attn"));
        self.fused.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("c_proj"));
        self.projection.visit(visitor);
        visitor.leave();
    }
}

/// The block's GELU MLP: up projection, activation, down projection.
struct FeedForward<E> {
    up: Linear<Tensor<E>>,
    activation: Gelu,
    down: Linear<Tensor<E>>,
}

impl<E: Elementary + From<f32>> FeedForward<E> {
    fn new(tape: &Tape<Tensor<E>>, activation: Gelu) -> Self {
        let zeros = |shape: [usize; 2]| Tensor::filled(shape, E::from(0.0));
        let bias = |extent: usize| Tensor::filled([extent], E::from(0.0));
        Self {
            up: Linear::new(tape, zeros([EMBED_DIM, 4 * EMBED_DIM]), bias(4 * EMBED_DIM)),
            activation,
            down: Linear::new(tape, zeros([4 * EMBED_DIM, EMBED_DIM]), bias(EMBED_DIM)),
        }
    }
}

impl<E: Elementary> Module<Tensor<E>> for FeedForward<E> {
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        input: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        let lifted = self.up.express(tape, input);
        let hidden = self.activation.express(tape, lifted);
        self.down.express(tape, hidden)
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.enter(Segment::Name("c_fc"));
        self.up.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("c_proj"));
        self.down.visit(visitor);
        visitor.leave();
    }
}

/// One pre-norm transformer block: attention and the MLP each read
/// their own normalization of the stream and add back into it.
struct Block<E> {
    attention_norm: LayerNorm<Tensor<E>>,
    attention: Attention<E>,
    hidden_norm: LayerNorm<Tensor<E>>,
    feed_forward: FeedForward<E>,
}

impl<E: Elementary + From<f32>> Block<E> {
    fn new(tape: &Tape<Tensor<E>>, mask: Symbol, scale: Symbol, activation: Gelu) -> Self {
        Self {
            attention_norm: layer_norm(tape),
            attention: Attention::new(tape, mask, scale),
            hidden_norm: layer_norm(tape),
            feed_forward: FeedForward::new(tape, activation),
        }
    }
}

impl<E: Elementary> Module<Tensor<E>> for Block<E> {
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        input: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        let attended = self
            .attention
            .express(tape, self.attention_norm.express(tape, input));
        let stream = input + attended;
        let lifted = self
            .feed_forward
            .express(tape, self.hidden_norm.express(tape, stream));
        stream + lifted
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.enter(Segment::Name("ln_1"));
        self.attention_norm.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("attn"));
        self.attention.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("ln_2"));
        self.hidden_norm.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("mlp"));
        self.feed_forward.visit(visitor);
        visitor.leave();
    }
}

/// Builds a layer norm with the conventional placeholder payloads and
/// the epsilon the checkpoint was trained with.
fn layer_norm<E: Elementary + From<f32>>(tape: &Tape<Tensor<E>>) -> LayerNorm<Tensor<E>> {
    LayerNorm::new(
        tape,
        Tensor::filled([EMBED_DIM], E::from(1.0)),
        Tensor::filled([EMBED_DIM], E::from(0.0)),
        Tensor::filled([], E::from(1e-5)),
    )
}

/// The whole model: token and position tables, the block stack, and
/// the final norm. The language-model head is not a child — it is the
/// embedding table tied, recorded transposed by the caller through
/// [`Gpt2::embeddings`].
pub struct Gpt2<E> {
    embeddings: Symbol,
    positions: Symbol,
    blocks: Sequential<Tensor<E>>,
    final_norm: LayerNorm<Tensor<E>>,
}

impl<E: Elementary + From<f32> + 'static> Gpt2<E> {
    /// Allocates the model's parameters with placeholder payloads, in
    /// visit order.
    ///
    /// Construction order is a contract: the emitted plan's leading
    /// arguments are the parameters in recording order, so recording
    /// them in visit order makes the positional snapshot exactly the
    /// emitted argument list.
    pub fn new(tape: &Tape<Tensor<E>>) -> Self {
        let embeddings = tape
            .parameter(Tensor::filled([VOCABULARY_LEN, EMBED_DIM], E::from(0.0)))
            .symbol();
        let positions = tape
            .parameter(Tensor::filled([POSITION_COUNT, EMBED_DIM], E::from(0.0)))
            .symbol();

        // The causal mask and the head scale are leaves shared by all
        // twelve blocks, like the GELU constants; leaves embed in the
        // plan as constants, so none of them join the argument list.
        let mask_elements: Vec<E> = (0..CONTEXT_LEN * CONTEXT_LEN)
            .map(|at| {
                if at % CONTEXT_LEN <= at / CONTEXT_LEN {
                    E::from(0.0)
                } else {
                    E::from(f32::NEG_INFINITY)
                }
            })
            .collect();
        let mask = tape
            .leaf(Tensor::new([CONTEXT_LEN, CONTEXT_LEN], mask_elements))
            .symbol();
        let scale = tape
            .leaf(Tensor::filled([], E::from(1.0 / (HEAD_DIM as f32).sqrt())))
            .symbol();
        let activation = Gelu::new(tape);

        let mut blocks = Sequential::new();
        for _ in 0..LAYER_COUNT {
            blocks = blocks.then(Block::new(tape, mask, scale, activation.clone()));
        }
        Self {
            embeddings,
            positions,
            blocks,
            final_norm: layer_norm(tape),
        }
    }

    /// Returns the symbol of the `[vocabulary, embed]` token table:
    /// the typed accessor the tied language-model head reads.
    pub fn embeddings(&self) -> Symbol {
        self.embeddings
    }
}

impl<E: Elementary> Module<Tensor<E>> for Gpt2<E> {
    /// Records the model over the embedded `[context, embed]` window:
    /// positions in, the block stack, the final norm.
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        input: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        let positions = tape.resolve(self.positions);
        let context = input.shape().axes()[0];
        let stream = input + positions.narrow(0, 0, context);
        self.final_norm
            .express(tape, self.blocks.express(tape, stream))
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.enter(Segment::Name("wte"));
        visitor.parameter("weights", self.embeddings);
        visitor.leave();
        visitor.enter(Segment::Name("wpe"));
        visitor.parameter("weights", self.positions);
        visitor.leave();
        visitor.enter(Segment::Name("h"));
        self.blocks.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("ln_f"));
        self.final_norm.visit(visitor);
        visitor.leave();
    }
}

/// Renders `path` as the checkpoint's tensor name. The tree mirrors
/// the released layout, so only the leaf spellings differ: the module
/// tier's `weights` and `scale` are the checkpoint's `weight`, and
/// its `shift` is the checkpoint's `bias`.
fn foreign_name(path: &Path) -> String {
    let segments = path.segments();
    let mut name = String::new();
    for (position, segment) in segments.iter().enumerate() {
        if position > 0 {
            name.push('.');
        }
        if position + 1 < segments.len() {
            name.push_str(&segment.to_string());
            continue;
        }
        let leaf = match segment {
            Segment::Name("weights") | Segment::Name("scale") => "weight",
            Segment::Name("shift") | Segment::Name("bias") => "bias",
            other => panic!("no checkpoint spelling for the leaf `{other}`"),
        };
        name.push_str(leaf);
    }
    name
}

/// Returns the state carrying the checkpoint: every parameter of
/// `model`'s tree restored by name, converted into the tree's element
/// type at the precision boundary.
///
/// This is the generic loader the module tier promises: the model
/// announces its own paths, [`foreign_name`] renders each as the
/// checkpoint's spelling, and [`named_restore`] does the rest —
/// missing tensors and shape mismatches fail loudly through the
/// restore's existing validation.
pub fn load<E: Elementary + From<f32>>(
    parameters: &Parameters<Tensor<E>>,
    model: &Gpt2<E>,
    weights: &Weights,
) -> Parameters<Tensor<E>> {
    let entries: Vec<(Path, Tensor<E>)> = named_parameters(model)
        .into_iter()
        .map(|(path, _)| {
            let payload = weights.tensor(&foreign_name(&path)).convert::<E>();
            (path, payload)
        })
        .collect();
    named_restore(parameters, model, entries)
}
