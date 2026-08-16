#!/usr/bin/env python3
"""Serves a textual StableHLO module through XLA: compile once, hold
the static arguments, then execute per request over binary pipes.

The serving sibling of `run-stablehlo-xla.py`, for callers that
execute one plan many times — a generation loop, a probe. Static
arguments (a model's parameters) load once from a binary file; each
request reads the dynamic trailing arguments from standard input and
writes the results to standard output, all raw little-endian `f32`.
Logs go to standard error, keeping standard output pure. The backend
follows jax's own selection: the default is the CPU, and
`JAX_PLATFORMS` picks another registered PJRT plugin.

Usage: serve-stablehlo-xla.py <module.mlir> <static.bin> <manifest.json>

`static.bin` holds the leading arguments in order, each as a `u32`
rank, `u32` extents, then the `f32` elements. The manifest names the
trailing per-request argument shapes:
`{"dynamic": [[256, 768], [1, 256]]}`.
"""

import json
import sys

import numpy as np

try:
    from jax.extend.backend import get_backend
except (AttributeError, ImportError):
    # The pre-0.5 spelling, the era Apple's jax-metal plugin pins.
    from jax.lib.xla_bridge import get_backend

with open(sys.argv[1]) as source:
    module_text = source.read()
with open(sys.argv[3]) as source:
    manifest = json.load(source)

static = []
with open(sys.argv[2], "rb") as source:
    data = source.read()
at = 0
while at < len(data):
    rank = int(np.frombuffer(data, dtype="<u4", count=1, offset=at)[0])
    at += 4
    shape = [
        int(extent)
        for extent in np.frombuffer(data, dtype="<u4", count=rank, offset=at)
    ]
    at += 4 * rank
    volume = int(np.prod(shape)) if shape else 1
    static.append(
        np.frombuffer(data, dtype="<f4", count=volume, offset=at).reshape(shape)
    )
    at += 4 * volume

backend = get_backend()
print(f"compiling on {backend.platform} ...", file=sys.stderr, flush=True)
try:
    from jaxlib import _jax

    devices = _jax.DeviceList(tuple(backend.local_devices()))
    executable = backend.compile_and_load(module_text, devices)
except ImportError:
    executable = backend.compile(module_text)


def device_buffer(argument):
    """Places one numpy argument on the backend's first device."""
    if hasattr(backend, "buffer_from_pyval"):
        # The pre-0.11 spelling, the era Apple's jax-metal plugin
        # pins — which may expose the method yet not implement it
        # (Metal answers UNIMPLEMENTED), so fall through on failure.
        try:
            return backend.buffer_from_pyval(argument)
        except Exception:
            pass
    import jax

    return jax.device_put(argument, backend.local_devices()[0])


static_buffers = [device_buffer(argument) for argument in static]
print("serving", file=sys.stderr, flush=True)

reader = sys.stdin.buffer
writer = sys.stdout.buffer
dynamic_shapes = manifest["dynamic"]
while True:
    dynamic = []
    for shape in dynamic_shapes:
        volume = int(np.prod(shape)) if shape else 1
        raw = reader.read(4 * volume)
        if not raw:
            sys.exit(0)
        dynamic.append(np.frombuffer(raw, dtype="<f4").reshape(shape))
    buffers = static_buffers + [device_buffer(argument) for argument in dynamic]
    for result in executable.execute(buffers):
        writer.write(np.asarray(result).astype("<f4").tobytes())
    writer.flush()
