//! TinyLlama (1.1B) as a module tree: the Llama architecture over the
//! released checkpoint's layout, expressed in topos's composition tier.
//!
//! Every struct here is an ordinary [`Module`] implementation — the
//! blocks are structs of bias-free projections and [`RmsNorm`]s around
//! a grouped-query attention module, the stack of twenty-two is a
//! [`Sequential`], and every Llama ingredient records from the public
//! op surface: rotary position embeddings are precomputed cosine and
//! sine leaves rotated by `narrow`/`neg`/`concat`, and the SwiGLU MLP
//! spells SiLU as `x / (1 + exp(-x))`. The tree's `visit` paths mirror
//! the checkpoint's own tensor names (`model.layers.{i}.self_attn.q_proj`,
//! `lm_head`, ...), so loading the pretrained weights is one
//! [`named_restore`] over the paths the model announces itself; the
//! adapter shrinks to the leaf spellings and the projection transpose
//! the module tier and the checkpoint disagree on.
//!
//! The tree is generic over the element type: the same structs record
//! the f32 model and the `Bf16` one, which is the genericity the
//! module design promises.

use std::marker::PhantomData;

use topos::checkpoint::named_restore;
use topos::{
    Differentiable, Elementary, Module, Parameters, Path, RmsNorm, Segment, Sequential, Symbol,
    Tape, Tensor, Value, Visitor, concat, named_parameters,
};

use crate::weights::Weights;

/// How many tokens of context the recorded graph attends over.
pub const CONTEXT_LEN: usize = 256;

/// How many dimensions the residual stream has.
pub const EMBED_DIM: usize = 2048;

/// How many query heads split the stream.
const HEAD_COUNT: usize = 32;

/// How many key/value heads the query heads share, in groups.
const KV_HEAD_COUNT: usize = 4;

/// How many dimensions each head reads and writes.
const HEAD_DIM: usize = EMBED_DIM / HEAD_COUNT;

/// How many query heads read each key/value head.
const GROUP_SIZE: usize = HEAD_COUNT / KV_HEAD_COUNT;

/// How many dimensions the MLP's hidden layer has.
const HIDDEN_DIM: usize = 5632;

/// How many transformer blocks the model stacks.
const LAYER_COUNT: usize = 22;

/// How many tokens the vocabulary holds.
pub const VOCABULARY_LEN: usize = 32000;

/// The rotary base the checkpoint was trained with.
const ROPE_BASE: f64 = 10000.0;

/// A bias-free linear map: one `[inputs, outputs]` parameter recorded
/// as a single matmul. The Llama architecture has no biases anywhere,
/// so `Linear`'s bias term would be a dead add on every projection.
struct Projection<E> {
    weights: Symbol,
    _marker: PhantomData<E>,
}

impl<E: Elementary + From<f32>> Projection<E> {
    /// Allocates the `[inputs, outputs]` parameter with a placeholder
    /// payload.
    fn new(tape: &Tape<Tensor<E>>, inputs: usize, outputs: usize) -> Self {
        Self {
            weights: tape
                .parameter(Tensor::filled([inputs, outputs], E::from(0.0)))
                .symbol(),
            _marker: PhantomData,
        }
    }
}

impl<E: Elementary> Module<Tensor<E>> for Projection<E> {
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        input: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        input.matmul(tape.resolve(self.weights))
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.parameter("weights", self.weights);
    }
}

/// Rotary position embeddings (Su et al., 2021) in the converted
/// checkpoint's rotate-half convention: the cosine and sine tables are
/// `[context, head]` leaves shared by every layer, precomputed because
/// they depend only on position and column — the record-time analog of
/// GPT-2's causal mask.
#[derive(Clone, Copy)]
struct Rope {
    cos: Symbol,
    sin: Symbol,
}

impl Rope {
    fn new<E: Elementary + From<f32>>(tape: &Tape<Tensor<E>>) -> Self {
        let half = HEAD_DIM / 2;
        let mut cosines = Vec::with_capacity(CONTEXT_LEN * HEAD_DIM);
        let mut sines = Vec::with_capacity(CONTEXT_LEN * HEAD_DIM);
        for position in 0..CONTEXT_LEN {
            for column in 0..HEAD_DIM {
                // Column `j` and column `j + half` share the frequency
                // `base^(-2 (j mod half) / head)`, the duplicated-halves
                // layout the rotate-half convention pairs with.
                let exponent = -2.0 * (column % half) as f64 / HEAD_DIM as f64;
                let angle = position as f64 * ROPE_BASE.powf(exponent);
                cosines.push(E::from(angle.cos() as f32));
                sines.push(E::from(angle.sin() as f32));
            }
        }
        Self {
            cos: tape
                .leaf(Tensor::new([CONTEXT_LEN, HEAD_DIM], cosines))
                .symbol(),
            sin: tape
                .leaf(Tensor::new([CONTEXT_LEN, HEAD_DIM], sines))
                .symbol(),
        }
    }

    /// Records the rotation of one head's `[context, head]` slice:
    /// `value * cos + rotate_half(value) * sin`, where `rotate_half`
    /// swaps the halves and negates the upper one.
    fn rotate<'tape, E: Elementary>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        value: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        let half = HEAD_DIM / 2;
        let cos = tape.resolve(self.cos);
        let sin = tape.resolve(self.sin);
        let flipped = concat(&[-value.narrow(1, half, half), value.narrow(1, 0, half)], 1);
        value * cos + flipped * sin
    }
}

/// Grouped-query causal self-attention: thirty-two query heads read
/// four shared key/value heads, every head a rank-2 `narrow` of the
/// separate projections, rotated by [`Rope`] before the scores.
struct Attention<E> {
    query: Projection<E>,
    key: Projection<E>,
    value: Projection<E>,
    output: Projection<E>,
    rope: Rope,
    mask: Symbol,
    scale: Symbol,
}

impl<E: Elementary + From<f32>> Attention<E> {
    /// Allocates the projections with placeholder payloads; `rope`,
    /// `mask`, and `scale` are leaves shared by every block.
    fn new(tape: &Tape<Tensor<E>>, rope: Rope, mask: Symbol, scale: Symbol) -> Self {
        Self {
            query: Projection::new(tape, EMBED_DIM, EMBED_DIM),
            key: Projection::new(tape, EMBED_DIM, KV_HEAD_COUNT * HEAD_DIM),
            value: Projection::new(tape, EMBED_DIM, KV_HEAD_COUNT * HEAD_DIM),
            output: Projection::new(tape, EMBED_DIM, EMBED_DIM),
            rope,
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
        let queries = self.query.express(tape, input);
        let keys = self.key.express(tape, input);
        let values = self.value.express(tape, input);

        // Each key/value head rotates and transposes once and serves
        // its whole group of query heads.
        let keyed: Vec<Value<'tape, Tensor<E>>> = (0..KV_HEAD_COUNT)
            .map(|group| {
                self.rope
                    .rotate(tape, keys.narrow(1, group * HEAD_DIM, HEAD_DIM))
                    .transpose()
            })
            .collect();

        let heads: Vec<Value<'tape, Tensor<E>>> = (0..HEAD_COUNT)
            .map(|head| {
                let group = head / GROUP_SIZE;
                let query = self
                    .rope
                    .rotate(tape, queries.narrow(1, head * HEAD_DIM, HEAD_DIM));
                let scores = query.matmul(keyed[group]);
                let weights = (scores * scale.broadcast_like(scores) + mask).softmax(1);
                weights.matmul(values.narrow(1, group * HEAD_DIM, HEAD_DIM))
            })
            .collect();
        self.output.express(tape, concat(&heads, 1))
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.enter(Segment::Name("q_proj"));
        self.query.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("k_proj"));
        self.key.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("v_proj"));
        self.value.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("o_proj"));
        self.output.visit(visitor);
        visitor.leave();
    }
}

/// The block's SwiGLU MLP (Shazeer, 2020): the gate's SiLU multiplies
/// the up projection elementwise before the down projection.
struct FeedForward<E> {
    gate: Projection<E>,
    up: Projection<E>,
    down: Projection<E>,
    one: Symbol,
}

impl<E: Elementary + From<f32>> FeedForward<E> {
    /// Allocates the projections with placeholder payloads; `one` is a
    /// scalar leaf shared by every block.
    fn new(tape: &Tape<Tensor<E>>, one: Symbol) -> Self {
        Self {
            gate: Projection::new(tape, EMBED_DIM, HIDDEN_DIM),
            up: Projection::new(tape, EMBED_DIM, HIDDEN_DIM),
            down: Projection::new(tape, HIDDEN_DIM, EMBED_DIM),
            one,
        }
    }
}

impl<E: Elementary> Module<Tensor<E>> for FeedForward<E> {
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        input: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        let one = tape.resolve(self.one);
        let gated = self.gate.express(tape, input);
        // SiLU as `x sigmoid(x)`, spelled `x / (1 + exp(-x))` from the
        // op surface.
        let activated = gated / ((-gated).exp() + one.broadcast_like(gated));
        self.down
            .express(tape, activated * self.up.express(tape, input))
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.enter(Segment::Name("gate_proj"));
        self.gate.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("up_proj"));
        self.up.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("down_proj"));
        self.down.visit(visitor);
        visitor.leave();
    }
}

/// One pre-norm transformer block: attention and the MLP each read
/// their own normalization of the stream and add back into it.
struct Block<E> {
    attention_norm: RmsNorm<Tensor<E>>,
    attention: Attention<E>,
    hidden_norm: RmsNorm<Tensor<E>>,
    feed_forward: FeedForward<E>,
}

impl<E: Elementary + From<f32>> Block<E> {
    fn new(tape: &Tape<Tensor<E>>, rope: Rope, mask: Symbol, scale: Symbol, one: Symbol) -> Self {
        Self {
            attention_norm: rms_norm(tape),
            attention: Attention::new(tape, rope, mask, scale),
            hidden_norm: rms_norm(tape),
            feed_forward: FeedForward::new(tape, one),
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
        visitor.enter(Segment::Name("input_layernorm"));
        self.attention_norm.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("self_attn"));
        self.attention.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("post_attention_layernorm"));
        self.hidden_norm.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("mlp"));
        self.feed_forward.visit(visitor);
        visitor.leave();
    }
}

/// Builds an RMS norm with the conventional placeholder payload and
/// the epsilon the checkpoint was trained with.
fn rms_norm<E: Elementary + From<f32>>(tape: &Tape<Tensor<E>>) -> RmsNorm<Tensor<E>> {
    RmsNorm::new(
        tape,
        Tensor::filled([EMBED_DIM], E::from(1.0)),
        Tensor::filled([], E::from(1e-5)),
    )
}

/// The whole model: the token table, the block stack, the final norm,
/// and the untied language-model head — Llama has no position table;
/// position enters through [`Rope`] inside every attention.
pub struct TinyLlama<E> {
    embeddings: Symbol,
    blocks: Sequential<Tensor<E>>,
    final_norm: RmsNorm<Tensor<E>>,
    head: Projection<E>,
}

impl<E: Elementary + From<f32> + 'static> TinyLlama<E> {
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

        // The rotary tables, the causal mask, the head scale, and the
        // SiLU's unit are leaves shared by all twenty-two blocks;
        // leaves embed in the plan as constants, so none of them join
        // the argument list.
        let rope = Rope::new(tape);
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
        let one = tape.leaf(Tensor::filled([], E::from(1.0))).symbol();

        let mut blocks = Sequential::new();
        for _ in 0..LAYER_COUNT {
            blocks = blocks.then(Block::new(tape, rope, mask, scale, one));
        }
        Self {
            embeddings,
            blocks,
            final_norm: rms_norm(tape),
            head: Projection::new(tape, EMBED_DIM, VOCABULARY_LEN),
        }
    }

    /// Returns the symbol of the `[vocabulary, embed]` token table:
    /// the typed accessor the loop-land embedding lookup reads.
    pub fn embeddings(&self) -> Symbol {
        self.embeddings
    }

    /// Records the untied head over the extracted `[1, embed]` row and
    /// returns the `[1, vocabulary]` logits.
    pub fn predict<'tape>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        last: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        self.head.express(tape, last)
    }
}

impl<E: Elementary> Module<Tensor<E>> for TinyLlama<E> {
    /// Records the model over the embedded `[context, embed]` window:
    /// the block stack, then the final norm.
    fn express<'tape>(
        &self,
        tape: &'tape Tape<Tensor<E>>,
        input: Value<'tape, Tensor<E>>,
    ) -> Value<'tape, Tensor<E>> {
        self.final_norm
            .express(tape, self.blocks.express(tape, input))
    }

    fn visit(&self, visitor: &mut dyn Visitor) {
        visitor.enter(Segment::Name("model"));
        visitor.enter(Segment::Name("embed_tokens"));
        visitor.parameter("weights", self.embeddings);
        visitor.leave();
        visitor.enter(Segment::Name("layers"));
        self.blocks.visit(visitor);
        visitor.leave();
        visitor.enter(Segment::Name("norm"));
        self.final_norm.visit(visitor);
        visitor.leave();
        visitor.leave();
        visitor.enter(Segment::Name("lm_head"));
        self.head.visit(visitor);
        visitor.leave();
    }
}

/// Renders `path` as the checkpoint's tensor name. The tree mirrors
/// the released layout, so only the leaf spellings differ: the module
/// tier's `weights` and `scale` are the checkpoint's `weight`.
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
            other => panic!("no checkpoint spelling for the leaf `{other}`"),
        };
        name.push_str(leaf);
    }
    name
}

/// Returns `tensor` with its two axes swapped, elementwise.
fn transposed(tensor: &Tensor<f32>) -> Tensor<f32> {
    let rows = tensor.shape().axes()[0];
    let columns = tensor.shape().axes()[1];
    let elements = tensor.to_vec();
    let mut flipped = vec![0.0; elements.len()];
    for row in 0..rows {
        for column in 0..columns {
            flipped[column * rows + row] = elements[row * columns + column];
        }
    }
    Tensor::new([columns, rows], flipped)
}

/// Returns the state carrying the checkpoint: every parameter of
/// `model`'s tree restored by name, converted into the tree's element
/// type at the precision boundary.
///
/// The checkpoint stores every `nn.Linear` weight as
/// `[outputs, inputs]`; topos's projections multiply as
/// `[inputs, outputs]`, so projection weights transpose once at this
/// boundary. The embedding table is a lookup, not a matmul, and stays
/// as released. A wrong choice here cannot pass silently:
/// [`named_restore`]'s shape validation rejects it.
pub fn load<E: Elementary + From<f32>>(
    parameters: &Parameters<Tensor<E>>,
    model: &TinyLlama<E>,
    weights: &Weights,
) -> Parameters<Tensor<E>> {
    let entries: Vec<(Path, Tensor<E>)> = named_parameters(model)
        .into_iter()
        .map(|(path, _)| {
            let name = foreign_name(&path);
            let released = weights.tensor(&name);
            let payload = if name.ends_with("proj.weight") || name == "lm_head.weight" {
                transposed(&released)
            } else {
                released
            };
            (path, payload.convert::<E>())
        })
        .collect();
    named_restore(parameters, model, entries)
}
