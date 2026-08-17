#!/usr/bin/env python3
"""Parses a textual StableHLO module; exit 0 means it parsed.

Tier-0 conformance for `Plan::emit_stablehlo`: any MLIR build with the
StableHLO dialect serves, and the smallest is a Python with `jax`
installed, whose bundled bindings this script uses. Point the
`TOPOS_STABLEHLO_VALIDATOR` environment variable at this script
(under such a Python) to enable the round-trip test in the suite:

    TOPOS_STABLEHLO_VALIDATOR="python3 tools/validate-stablehlo.py" cargo test
"""

import sys

from jax._src.interpreters.mlir import make_ir_context
from jaxlib.mlir.ir import Module

with open(sys.argv[1]) as source:
    text = source.read()
with make_ir_context():
    Module.parse(text)
print("parsed OK")
