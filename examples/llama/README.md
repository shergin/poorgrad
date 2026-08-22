# Running the Llama family on topos

This example generates text with released Llama-family weights — the
whole architecture recorded on the tape from the existing op surface:
no new opcodes, no ML dependency, no Python anywhere. Where the GPT-2
example proved the op surface is done for transformers, this one
proves it covers the modern Llama recipe — RMS normalization, rotary
position embeddings, grouped-query attention, and a SwiGLU MLP — and
that the architecture scales as data: one module tree records both
TinyLlama 1.1B and Llama 2 7B, the model picked by a `Family`
descriptor of dimensions and download URLs.

## Quick start

```sh
# TinyLlama 1.1B (default): a 4.1 GB download, quick to a first token.
cargo run --release --features accelerate --example llama -- "Once upon a time"

# Llama 2 7B: a 13.5 GB download, bf16 recommended (see below).
cargo run --release --features accelerate --example llama -- "Once upon a time" 40 bf16 llama2
```

The first run of each model downloads and caches its artifacts from
Hugging Face into `~/.cache/topos/<model>` — shared by every checkout
and worktree, never seen by git. Llama 2 7B comes through an ungated
mirror of the converted layout (`NousResearch/Llama-2-7b-hf`); the
original `meta-llama` repository is gated.

The `accelerate` feature is the right build on a Mac; `simd` is the
portable rung elsewhere. The default build works too, just slower —
the products fall to the safe slice path.

## Arguments

```sh
cargo run --release --features accelerate --example llama -- [PROMPT] [COUNT] [ENGINE] [MODEL]
```

| position | meaning | default |
|---|---|---|
| 1 | the prompt | `The library of the poor holds one book` |
| 2 | how many tokens to generate | `40` |
| 3 | the engine: `tape` or `bf16` | `tape` |
| 4 | the model: `tinyllama` or `llama2` | `tinyllama` |

The recorded graph attends over a fixed 256-token context, so the
prompt plus the generation count must fit inside it; the example
asserts this up front. Sampling is temperature 0.9 with top-k 40
under a fixed seed, so a given prompt, count, engine, and model
reproduce their text exactly.

`tape` runs the compiled plan on topos's own interpreter over f32.
`bf16` records the identical module tree over `Tensor<Bf16>` and
runs it on the same interpreter: the model code is generic over the
element type, so the half-precision variant is one type argument,
with the checkpoint converted at the precision boundary and every
matmul accumulating in `f32` by the payload's contract. Half the
memory, its own (coherent) text — bf16 rounding is a different
model, not a noisy copy of the f32 one. At 7B the memory half is
the point: bf16 keeps the resident parameters near 13 GB where f32
wants 27, which is why `bf16` is the recommended engine for `llama2`
on a 32 GB machine.

Measured on an M1 Pro (32 GB) with `accelerate`, same prompt and
seed:

| model | engine | ms/token | warm start to first token |
|---|---|---|---|
| `tinyllama` | `tape` | 1490 | ~10 s |
| `tinyllama` | `bf16` | 2000 | ~14 s |
| `llama2` | `bf16` | 7000-8000 | ~95 s |

With no KV cache — the plan re-runs the whole 256-token window every
step — seconds per token is the honest cost of a whole Llama on the
interpreter. The 7B warm start is dominated by reading and widening
the 13.5 GB checkpoint from disk.

## How it works

The model lives in [`model.rs`](model.rs) as a module tree: pre-norm
blocks — each a struct of bias-free projections and `RmsNorm`s
around a grouped-query attention module — stacked in a `Sequential`,
the whole tree generic over the element type and built from a
[`family.rs`](family.rs) descriptor's dimensions. Each Llama
ingredient is a few lines over the public op surface:

- **Rotary position embeddings** are precomputed cosine and sine
  leaves — they depend only on position and column, and the context
  is fixed at record time, so they embed as constants the way GPT-2's
  causal mask does. The rotation itself is `narrow`, `neg`, `concat`,
  and elementwise arithmetic.
- **Grouped-query attention** slices per-head rank-2 views by
  `narrow` out of the separate query/key/value projections; each
  key/value head rotates and transposes once and serves its whole
  group of query heads. TinyLlama shares 4 key/value heads among 32
  query heads; Llama 2 7B is the one-head-per-group special case —
  plain multi-head attention.
- **The SwiGLU MLP** spells SiLU as `x / (1 + exp(-x))` with a shared
  scalar-one leaf, the same way the GPT-2 example hand-rolls its GELU.
- **RMS normalization** is the crate's own `RmsNorm` facade with the
  checkpoints' epsilon.

The tree's `visit` paths mirror the checkpoints' own tensor names
(`model.layers.{i}.self_attn.q_proj`, `lm_head`, ...), so loading the
pretrained weights is one `named_restore` — but the iteration is the
checkpoint's, not the tree's: the safetensors reader beside this file
streams tensors shard by shard, so only one shard's bytes are ever
resident, and each tensor converts to the tree's element type as it
arrives. At 7B that streaming is the difference between a ~20 GB load
peak and one over 50. Tensors the tree never asked for (older
conversions ship per-layer `rotary_emb.inv_freq` tables) are skipped;
missing or misshapen ones fail loudly through the restore's own
validation. The checkpoints store every `nn.Linear` weight as
`[outputs, inputs]` while topos's projections multiply as
`[inputs, outputs]`, so projection weights transpose once at the load
boundary. The reader widens f32, bf16, and f16 elements — TinyLlama
ships f32, Llama 2 f16.

The prompt goes through the family's SentencePiece-style BPE (the
metaspace convention, ranked merges, byte fallback), hand-rolled in
[`tokenizer.rs`](tokenizer.rs) and verified id-for-id against the
reference implementation; only the JSON syntax of `tokenizer.json` is
read by `serde_json`. The tokenizer round-trips the prompt on every
run as a self-check.

## Troubleshooting

- **The download fails.** The example shells out to `curl`; place the
  model's files under `~/.cache/topos/<model>` by hand and it will
  use them as-is.
- **Memory.** TinyLlama settles around 4.5 GB resident (f32). Llama 2
  7B peaks near 20 GB during the bf16 load and settles near 14 GB
  while generating; the f32 engine wants roughly twice that and is
  not recommended under 48 GB.

What a *fast* Llama would still need is not in this example by
design: a KV cache inside the fixed-shape plan and quantized
payloads are engine-tier conversations, not example-tier ones. The
backend ladder this example rides — and the emission road the GPT-2
example takes further — is told in
[ACCELERATION.md](../../ACCELERATION.md).
