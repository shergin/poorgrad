# Running TinyLlama on topos

This example generates text with the released TinyLlama (1.1B) base
weights — the Llama architecture end to end, recorded on the tape
from the existing op surface: no new opcodes, no ML dependency, no
Python anywhere. Where the GPT-2 example proved the op surface is
done for transformers, this one proves it covers the modern Llama
family: RMS normalization, rotary position embeddings, grouped-query
attention, and a SwiGLU MLP all record from ops that already existed.

## Quick start

```sh
cargo run --release --features accelerate --example tinyllama -- "Once upon a time"
```

The first run downloads and caches two artifacts from Hugging Face
(`model.safetensors` at 4.1 GB, `tokenizer.json`) into
`~/.cache/topos/tinyllama` — shared by every checkout and worktree,
never seen by git — then loads the checkpoint, records ~16000 nodes,
compiles the sampling plan, and generates. Every later run reaches
the first token in about ten seconds.

The `accelerate` feature is the right build on a Mac; `simd` is the
portable rung elsewhere. The default build works too, just slower —
the products fall to the safe slice path.

## Arguments

```sh
cargo run --release --features accelerate --example tinyllama -- [PROMPT] [COUNT] [ENGINE]
```

| position | meaning | default |
|---|---|---|
| 1 | the prompt | `The library of the poor holds one book` |
| 2 | how many tokens to generate | `40` |
| 3 | the engine: `tape` or `bf16` | `tape` |

The recorded graph attends over a fixed 256-token context, so the
prompt plus the generation count must fit inside it; the example
asserts this up front. Sampling is temperature 0.9 with top-k 40
under a fixed seed, so a given prompt, count, and engine reproduce
their text exactly.

`tape` runs the compiled plan on topos's own interpreter over f32.
`bf16` records the identical module tree over `Tensor<Bf16>` and
runs it on the same interpreter: the model code is generic over the
element type, so the half-precision variant is one type argument,
with the checkpoint converted at the precision boundary and every
matmul accumulating in `f32` by the payload's contract. Half the
memory, its own (coherent) text — bf16 rounding is a different
model, not a noisy copy of the f32 one.

Measured on an M1 Pro with `accelerate`, same prompt and seed:

| engine | ms/token |
|---|---|
| `tape` | 1490 |
| `bf16` | 2000 |

At 1.1B parameters and no KV cache — the plan re-runs the whole
256-token window every step — one to two seconds per token is the
honest cost of a whole Llama on the interpreter; the GPT-2 example's
124M runs the same design nine times lighter.

## How it works

The model lives in [`model.rs`](model.rs) as a module tree:
twenty-two pre-norm blocks — each a struct of bias-free projections
and `RmsNorm`s around a grouped-query attention module — stacked in
a `Sequential`, the whole tree generic over the element type. Each
Llama ingredient is a few lines over the public op surface:

- **Rotary position embeddings** are precomputed cosine and sine
  leaves — they depend only on position and column, and the context
  is fixed at record time, so they embed as constants the way GPT-2's
  causal mask does. The rotation itself is `narrow`, `neg`, `concat`,
  and elementwise arithmetic.
- **Grouped-query attention** slices per-head rank-2 views by
  `narrow` out of the separate query/key/value projections; each of
  the four key/value heads rotates and transposes once and serves its
  group of eight query heads.
- **The SwiGLU MLP** spells SiLU as `x / (1 + exp(-x))` with a shared
  scalar-one leaf, the same way the GPT-2 example hand-rolls its GELU.
- **RMS normalization** is the crate's own `RmsNorm` facade with the
  checkpoint's epsilon.

The tree's `visit` paths mirror the checkpoint's own tensor names
(`model.layers.{i}.self_attn.q_proj`, `lm_head`, ...), so loading the
pretrained weights is one `named_restore` over the paths the model
announces itself. The checkpoint stores every `nn.Linear` weight as
`[outputs, inputs]` while topos's projections multiply as
`[inputs, outputs]`, so projection weights transpose once at the load
boundary — a wrong choice cannot pass silently, because the restore's
shape validation rejects it. The safetensors reader beside this file
widens f32, bf16, and f16 elements, so the loader is not tied to this
release's dtype.

The prompt goes through TinyLlama's SentencePiece-style BPE (the
metaspace convention, ranked merges, byte fallback), hand-rolled in
[`tokenizer.rs`](tokenizer.rs); only the JSON syntax of
`tokenizer.json` is read by `serde_json`. The tokenizer round-trips
the prompt on every run as a self-check.

## Troubleshooting

- **The download fails.** The example shells out to `curl`; place the
  two files under `~/.cache/topos/tinyllama` by hand and it will use
  them as-is.
- **Memory.** Loading holds the checkpoint plus the recorded
  parameters — around 12 GB at peak for f32, settling to about 4.5 GB
  while generating; `bf16` halves the settled figure.

What a *fast* Llama would still need is not in this example by
design: a KV cache inside the fixed-shape plan and quantized
payloads are engine-tier conversations, not example-tier ones. The
backend ladder this example rides — and the emission road the GPT-2
example takes further — is told in
[ACCELERATION.md](../../ACCELERATION.md).
