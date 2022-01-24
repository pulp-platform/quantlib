from functools import partial
from ...util.tracing import LeafTracer, custom_symbolic_trace
from ..pact.pact_util import PACT_OPS
from quantlib.algorithms.bb.bb_ops import *

__all__ = ["BB_OPS",
           "BBTracer",
           "BB_symbolic_trace"]

BB_OPS = set([BBAct,
              BBConv2d,
              BBLinear])

BBTracer = LeafTracer(leaf_types=list(BB_OPS | PACT_OPS))
BB_symbolic_trace = partial(custom_symbolic_trace, tracer=BBTracer)
