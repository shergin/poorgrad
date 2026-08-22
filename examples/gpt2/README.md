# Running GPT-2 on topos

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
`~/.cache/topos/gpt2` — shared by every checkout and worktree,
never seen by git — then loads the checkpoint, records ~5600 nodes
(the full-context expression and its one-token decode twin on one
tape), compiles a plan, and generates. Every later run starts in
about a second.

The `accelerate` feature is the right build on a Mac (about
18 ms/token on an M1 Pro through the decode plan); `simd` is the
portable rung elsewhere. The default build works too, just slower —
the products fall to the safe slice path.

## Arguments

```sh
cargo run --release --features accelerate --example gpt2 -- [PROMPT] [COUNT] [ENGINE]
```

| position | meaning | default |
|---|---|---|
| 1 | the prompt | `The library of the poor holds one book` |
| 2 | how many tokens to generate | `40` |
| 3 | the engine: `tape`, `full`, `bf16`, or `xla` | `tape` |

The recorded graph attends over a fixed 256-token context, so the
prompt plus the generation count must fit inside it; the example
asserts this up front. Sampling is temperature 0.9 with top-k 40
under a fixed seed, so a given prompt, count, and engine reproduce
their text exactly.

## The four engines

**`tape`** generates through the one-token decode plan on topos's
own interpreter — a KV cache with no new engine concept
(`notes/carry.md`). Each layer's keys and values live in
capacity-shaped caches that are ordinary per-run inputs; a step
feeds the carry plus ~4 KB of transients (the embedded token row, a
position one-hot, a mask row), appends the new rows by `scatter`,
and reads the logits plus the updated caches back. Prefill is
token-by-token through the same plan.

**`full`** is the pre-cache loop kept as the baseline: the whole
256-token window re-embedded and re-run for every token through the
full-context plan. Same seed, same text — the decode plan
reproduces it token for token, which is the cross-graph check the
two expressions exist to make.

**`bf16`** records the identical module tree over `Tensor<Bf16>`
and decodes on the same interpreter: the model code is generic over
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

TOPOS_XLA_PYTHON="$HOME/jax-venv/bin/python3" \
  cargo run --release --features accelerate --example gpt2 -- "Once upon a time" 40 xla
```

`TOPOS_XLA_PYTHON` names the Python (default `python3`), and
`JAX_PLATFORMS` picks the XLA backend the jax way. The first token
waits a few seconds while the server compiles the module — a warmup
step keeps that out of the per-token figure — and the server's log
goes to standard error.

Measured on an M1 Pro, same prompt and seed:

| engine | ms/token | output |
|---|---|---|
| `tape` (+`accelerate`, decode plan) | 18 | the reference text |
| `full` (+`accelerate`) | 193 | identical to the tape's |
| `bf16` (+`accelerate`, decode plan) | 31 | its own text, by rounding |
| `xla` on XLA-CPU (full-context plan) | 132 | identical to the tape's |
| `xla` on Metal (`jax-metal`) | 26 | wrong — see below |

That the decode plan, the full-context plan, and XLA-CPU produce
identical text is the point of having them all: the same function,
two graphs and two executors, agreeing token for token — and an
in-crate test (`decode_carry_matches_the_full_context_plan_bitwise`)
pins the two graphs bit for bit at toy scale.

## The Metal cautionary tale

Apple ships an experimental PJRT plugin, `jax-metal`, pinned to the
jax 0.4.26 era:

```sh
python3 -m venv ~/jax-metal-venv
~/jax-metal-venv/bin/pip install "jax==0.4.26" "jaxlib==0.4.26" jax-metal

JAX_PLATFORMS=METAL TOPOS_XLA_PYTHON="$HOME/jax-metal-venv/bin/python3" \
  cargo run --release --features accelerate --example gpt2 -- "Once upon a time" 40 xla
```

It runs this module at 26 ms/token on the GPU — and generates
confident nonsense. The plugin passes topos's small conformance
modules but miscomputes this one, and the verdict is provable rather
than a matter of taste because three independent implementations —
the tape, compiled XLA-CPU, and the StableHLO reference
interpreter — agree with each other and it does not. Run it once;
it is the whole conformance story in one command.

## How it works

The model lives in [`model.rs`](model.rs) as a module tree: twelve
pre-norm blocks — each a struct of `Linear`s and `LayerNorm`s
around a custom attention module — stacked in a plain `Vec`, with
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
accessor (the decode step spells it `(wte . row^T)^T`, so no run
materializes the transposed table). Forward-only plans compiled
once serve every step of every engine.

Beside `express`, each struct records a one-token decode step
(`express_decode`): the caches arrive as inputs, the new key and
value rows land by `scatter` over the position one-hot (the row
being still zero, the append is a pure add), and the same one-hot
gathers the position embedding row. The loop's only state is the
`Carry` in [`main.rs`](main.rs) — the pending cache feeds, a plain
caller-owned value advanced from each run's cache outputs, so two
divergent continuations of one prefill are one `Carry` clone
apart.

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
  the three files under `~/.cache/topos/gpt2` by hand and it
  will use them as-is.
- **The XLA server does not start.** The named Python cannot import
  `jax` — check `TOPOS_XLA_PYTHON` and the venv. Its compile log
  and any traceback go to standard error.
- **Memory.** Loading holds the checkpoint plus the recorded
  parameters — a few gigabytes at peak; any machine that runs a
  browser is fine.

Where this sits in the larger design — emission, the conformance
tiers, and the measured serving numbers — is told in
[ACCELERATION.md](../../ACCELERATION.md).
