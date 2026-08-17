#!/usr/bin/env python3
"""Executes a textual StableHLO module through the reference
interpreter and prints its results.

Tier-1 conformance for `Plan::emit_stablehlo`: the reference
interpreter is the StableHLO specification's executable semantics, so
agreement here is semantic conformance, not just syntax. Any Python
with `jax` installed serves, through its bundled bindings. Point the
`TOPOS_STABLEHLO_EVALUATOR` environment variable at this script to
enable the execution test in the suite:

    TOPOS_STABLEHLO_EVALUATOR="python3 tools/eval-stablehlo.py" cargo test

Usage: eval-stablehlo.py <module.mlir> <arguments.txt>

Each argument line and each printed result line is one tensor: the
`x`-joined extents (`-` for rank 0), then the elements in row-major
order as plain decimals. The element type of each argument is read
from the module's own `@main` signature, so `f32`, `f64`, `bf16`,
and `f16` modules all feed correctly; arguments cross the boundary
as parsed dense literals, which is the one path the MLIR bindings
support for every float width.
"""

import re
import struct
import sys

from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir.dialects import stablehlo
from jaxlib.mlir.ir import Attribute, DenseFPElementsAttr, Module, ShapedType


def bits_of(value, element):
    """The IEEE bit pattern of `value` in `element`, for the hex
    literal MLIR requires for non-finite values."""
    single = struct.unpack("<I", struct.pack("<f", value))[0]
    if element == "bf16":
        return single >> 16, 4
    if element == "f16":
        half = struct.unpack("<H", struct.pack("<e", value))[0]
        return half, 4
    if element == "f32":
        return single, 8
    return struct.unpack("<Q", struct.pack("<d", value))[0], 16


def value_of(token, element):
    """The float a printed element token denotes: a plain decimal, or
    a bit-pattern hex for the non-finite values."""
    if not token.startswith("0x"):
        return float(token)
    bits = int(token, 16)
    if element == "bf16":
        return struct.unpack("<f", struct.pack("<I", bits << 16))[0]
    if element == "f16":
        return struct.unpack("<e", struct.pack("<H", bits))[0]
    if element == "f32":
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    return struct.unpack("<d", struct.pack("<Q", bits))[0]


def literal(value, element):
    """One element of a dense literal in `element`."""
    parsed = float(value)
    if parsed == float("inf") or parsed == float("-inf") or parsed != parsed:
        bits, width = bits_of(parsed, element)
        return f"0x{bits:0{width}X}"
    return value


def nested(values, dimensions):
    """The bracket nesting MLIR's dense literal syntax requires."""
    if not dimensions:
        return values[0]
    if len(dimensions) == 1:
        return "[" + ", ".join(values) + "]"
    stride = len(values) // dimensions[0]
    rows = (
        nested(values[row * stride : (row + 1) * stride], dimensions[1:])
        for row in range(dimensions[0])
    )
    return "[" + ", ".join(rows) + "]"


def attribute_values(attribute, element):
    """The elements of a dense result attribute in row-major order.

    Iteration converts directly for `f32`/`f64`; the narrower floats
    have no binding-level accessor, so their values are read back from
    the attribute's own textual form, splats expanded.
    """
    shape = ShapedType(attribute.type).shape
    volume = 1
    for extent in shape:
        volume *= extent
    try:
        return [float(value) for value in attribute]
    except TypeError:
        text = str(attribute)
        payload = re.match(r"dense<(.*)> : tensor<[^<>]*>$", text, re.DOTALL).group(1)
        tokens = re.findall(
            r"0x[0-9A-Fa-f]+|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", payload
        )
        values = [value_of(token, element) for token in tokens]
        if len(values) == 1 and volume > 1:
            values = values * volume
        return values


with open(sys.argv[1]) as source:
    module_text = source.read()

signature = re.search(r"@main\((.*?)\)\s*->", module_text, re.DOTALL)
if signature is None:
    raise SystemExit("the module has no @main signature")
elements = [
    tensor_type.split("x")[-1]
    for tensor_type in re.findall(r"tensor<([^<>]*)>", signature.group(1))
]

argument_texts = []
with open(sys.argv[2]) as source:
    for line in source.read().splitlines():
        if not line.strip():
            continue
        dimensions_text, _, values_text = line.partition(" ")
        dimensions = (
            [] if dimensions_text == "-" else [int(d) for d in dimensions_text.split("x")]
        )
        element = elements[len(argument_texts)]
        values = [literal(value, element) for value in values_text.split()]
        tensor_type = "x".join([str(extent) for extent in dimensions] + [element])
        argument_texts.append(f"dense<{nested(values, dimensions)}> : tensor<{tensor_type}>")

with make_ir_context():
    module = Module.parse(module_text)
    results = stablehlo.eval_module(
        module, [Attribute.parse(text) for text in argument_texts]
    )
    for result in results:
        attribute = DenseFPElementsAttr(result)
        shaped = ShapedType(attribute.type)
        element = str(shaped.element_type)
        dimensions = "x".join(str(extent) for extent in shaped.shape) or "-"
        values = attribute_values(attribute, element)
        print(dimensions, " ".join(repr(float(value)) for value in values))
