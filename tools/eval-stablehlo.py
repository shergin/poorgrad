#!/usr/bin/env python3
"""Executes a textual StableHLO module through the reference
interpreter and prints its results.

Tier-1 conformance for `Plan::emit_stablehlo`: the reference
interpreter is the StableHLO specification's executable semantics, so
agreement here is semantic conformance, not just syntax. Any Python
with `jax` installed serves, through its bundled bindings. Point the
`POORGRAD_STABLEHLO_EVALUATOR` environment variable at this script to
enable the execution test in the suite:

    POORGRAD_STABLEHLO_EVALUATOR="python3 tools/eval-stablehlo.py" cargo test

Usage: eval-stablehlo.py <module.mlir> <arguments.txt>

Each argument line and each printed result line is one `f32` tensor:
the `x`-joined extents (`-` for rank 0), then the elements in
row-major order.
"""

import sys

import numpy as np
from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.ir import DenseElementsAttr, DenseFPElementsAttr, Module, ShapedType

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

with make_ir_context():
    module = Module.parse(module_text)
    results = stablehlo.eval_module(
        module, [DenseElementsAttr.get(argument) for argument in arguments]
    )
    for result in results:
        attribute = DenseFPElementsAttr(result)
        shape = ShapedType(attribute.type).shape
        dimensions = "x".join(str(extent) for extent in shape) or "-"
        print(dimensions, " ".join(repr(float(value)) for value in attribute))
