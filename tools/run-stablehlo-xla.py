#!/usr/bin/env python3
"""Compiles and executes a textual StableHLO module through XLA and
prints its results.

The compiled sibling of `eval-stablehlo.py`, speaking the same line
protocol, so it drops into the same suite seam: pointing
`POORGRAD_STABLEHLO_EVALUATOR` at this script runs tier-1 conformance
against a real XLA backend instead of the reference interpreter. The
backend follows jax's own selection: the default is the CPU, and
`JAX_PLATFORMS` picks another registered PJRT plugin (for example
`JAX_PLATFORMS=METAL` with `jax-metal` installed).

Usage: run-stablehlo-xla.py <module.mlir> <arguments.txt>

Each argument line and each printed result line is one `f32` tensor:
the `x`-joined extents (`-` for rank 0), then the elements in
row-major order.
"""

import sys

import numpy as np

try:
    from jax.extend.backend import get_backend
except (AttributeError, ImportError):
    # The pre-0.5 spelling, the era Apple's jax-metal plugin pins.
    from jax.lib.xla_bridge import get_backend

with open(sys.argv[1]) as source:
    module_text = source.read()

arguments = []
with open(sys.argv[2]) as source:
    for line in source.read().splitlines():
        if not line.strip():
            continue
        dimensions_text, _, values_text = line.partition(" ")
        dimensions = (
            [] if dimensions_text == "-" else [int(d) for d in dimensions_text.split("x")]
        )
        values = np.array([float(v) for v in values_text.split()], dtype=np.float32)
        arguments.append(values.reshape(dimensions))

backend = get_backend()
try:
    # The current PJRT client API wants the device list spelled out.
    from jaxlib import _jax

    devices = _jax.DeviceList(tuple(backend.local_devices()))
    executable = backend.compile_and_load(module_text, devices)
except ImportError:
    # Older clients — the era Apple's jax-metal plugin pins — take the
    # module alone.
    executable = backend.compile(module_text)
buffers = [backend.buffer_from_pyval(argument) for argument in arguments]
results = executable.execute(buffers)
for result in results:
    array = np.asarray(result)
    dimensions = "x".join(str(extent) for extent in array.shape) or "-"
    print(dimensions, " ".join(repr(float(value)) for value in array.ravel()))
