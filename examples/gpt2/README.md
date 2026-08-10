# Running GPT-2 on poorgrad

This example generates text with OpenAI's released GPT-2 (124M)
weights, the whole model recorded on the tape from the existing op
surface — no new opcodes, no ML dependency, no Python in the loop
unless you opt into the XLA engine. It exists to prove a claim: the
op surface is done for transformers, and the same compiled plan can
run at home or be written down as StableHLO and served by an
industrial compiler, with the two engines checking each other.

## Quick start

```sh
cargo run --release --features accelerate --example gpt2 -- "Once upon a time"
```

The first run downloads and caches three artifacts from Hugging Face
(`model.safetensors` at 548 MB, `vocab.json`, `merges.txt`) into
`~/.cache/poorgrad/gpt2` — shared by every checkout and worktree,
never seen by git — then loads the checkpoint, records ~2800 nodes,
compiles the sampling plan, and generates. Every later run starts in
about a second.

The `accelerate` feature is the right build on a Mac (about
195 ms/token on an M1 Pro); `simd` is the portable rung elsewhere.
The default build works too, just slower — the products fall to the
safe slice path.

## Arguments

```sh
cargo run --release --features accelerate --example gpt2 -- [PROMPT] [COUNT] [ENGINE]
```

| position | meaning | default |
|---|---|---|
| 1 | the prompt | `The library of the poor holds one book` |
| 2 | how many tokens to generate | `40` |
| 3 | the engine: `tape`, `bf16`, or `xla` | `tape` |

The recorded graph attends over a fixed 256-token context, so the
prompt plus the generation count must fit inside it; the example
asserts this up front. Sampling is temperature 0.9 with top-k 40
under a fixed seed, so a given prompt, count, and engine reproduce
their text exactly.

## The three engines

**`tape`** runs the plan on poorgrad's own interpreter — the
oracle. Everything happens in-process; the per-step feeds are the
embedded token window and the prediction row's one-hot extraction,
so generation never regrows the tape.

**`bf16`** records the identical module tree over `Tensor<Bf16>`
and runs it on the same interpreter: the model code is generic over
the element type, so the half-precision variant is one type
argument, with the checkpoint converted at the precision boundary
and every matmul accumulating in `f32` by the payload's contract.
Half the memory, its own (coherent) text — bf16 rounding is a
different model, not a noisy copy of the f32 one.

**`xla`** emits the f32 plan as a textual StableHLO module and
holds a serving process, [`tools/serve-stablehlo-xla.py`](../../tools/serve-stablehlo-xla.py):
compile once, keep the 124M parameters resident (they cross the
boundary once, as a binary sidecar), and answer each step over raw
`f32` pipes — a step ships the ~787 KB embedded window, not the
module. It needs a Python with `jax` installed; current jax wheels
want Python 3.10-3.13:

```sh
python3 -m venv ~/jax-venv
~/jax-venv/bin/pip install jax

POORGRAD_XLA_PYTHON="$HOME/jax-venv/bin/python3" \
  cargo run --release --features accelerate --example gpt2 -- "Once upon a time" 40 xla
```

`POORGRAD_XLA_PYTHON` names the Python (default `python3`), and
`JAX_PLATFORMS` picks the XLA backend the jax way. The first token
waits a few seconds while the server compiles the module — a warmup
step keeps that out of the per-token figure — and the server's log
goes to standard error.

Measured on an M1 Pro, same prompt and seed:

| engine | ms/token | output |
|---|---|---|
| `tape` (+`accelerate`) | 194 | the reference text |
| `bf16` (+`accelerate`) | 341 | its own text, by rounding |
| `xla` on XLA-CPU | 132 | identical to the tape's |
| `xla` on Metal (`jax-metal`) | 26 | wrong — see below |

That the tape and XLA-CPU produce identical text is the point of
having both: the same function, one interpreter and one industrial
compiler, agreeing token for token.

## The Metal cautionary tale

Apple ships an experimental PJRT plugin, `jax-metal`, pinned to the
jax 0.4.26 era:

```sh
python3 -m venv ~/jax-metal-venv
~/jax-metal-venv/bin/pip install "jax==0.4.26" "jaxlib==0.4.26" jax-metal

JAX_PLATFORMS=METAL POORGRAD_XLA_PYTHON="$HOME/jax-metal-venv/bin/python3" \
  cargo run --release --features accelerate --example gpt2 -- "Once upon a time" 40 xla
```

It runs this module at 26 ms/token on the GPU — and generates
confident nonsense. The plugin passes poorgrad's small conformance
modules but miscomputes this one, and the verdict is provable rather
than a matter of taste because three independent implementations —
the tape, compiled XLA-CPU, and the StableHLO reference
interpreter — agree with each other and it does not. Run it once;
it is the whole conformance story in one command.

## How it works

The model lives in [`model.rs`](model.rs) as a module tree: twelve
pre-norm blocks — each a struct of `Linear`s and `LayerNorm`s
around a custom attention module — stacked in a `Sequential`, with
the whole tree generic over the element type (that genericity is
the `bf16` engine). Attention slices per-head rank-2 views by
`narrow` out of one fused query-key-value `Linear`, joins the heads
by `concat`, and adds the causal mask as an additive `0 / -inf`
leaf; the GELU MLP's tanh-approximation constants ride as scalar
leaves. The token embedding lookup is loop-land data preparation —
a row copy from the table, like makemore's context assembly — so
the plan's input is the embedded window and the vocabulary-sized
one-hot never crosses any boundary. The tied language-model head is
the embedding table transposed, read through the module's typed
accessor. One forward-only plan serves every step of every engine.

The tree's `visit` paths mirror the checkpoint's own tensor names
(`h.{i}.attn.c_attn`, `ln_f`, ...), so loading the pretrained
weights is one `named_restore` over the paths the model announces
itself: the tree allocates with placeholder payloads, each path is
rendered as the checkpoint's spelling (only the leaf names differ),
and the restore builds the generation that carries the weights —
missing tensors and shape mismatches fail loudly through the
restore's own validation.

The safetensors file itself is read by a hand-rolled reader (an
8-byte header length, a JSON header, raw `f32` data) and the prompt
through GPT-2's byte-level BPE (pretokenizer, byte-to-unicode
table, ranked merges), both living beside this file. Only the JSON
syntax in each is read by `serde_json`; every format and algorithm
around it is in view. The tokenizer round-trips the prompt on every run as a
self-check.

## Troubleshooting

- **The download fails.** The example shells out to `curl`; place
  the three files under `~/.cache/poorgrad/gpt2` by hand and it
  will use them as-is.
- **The XLA server does not start.** The named Python cannot import
  `jax` — check `POORGRAD_XLA_PYTHON` and the venv. Its compile log
  and any traceback go to standard error.
- **Memory.** Loading holds the checkpoint plus the recorded
  parameters — a few gigabytes at peak; any machine that runs a
  browser is fine.

Where this sits in the larger design — emission, the conformance
tiers, and the measured serving numbers — is told in
[ACCELERATION.md](../../ACCELERATION.md).
