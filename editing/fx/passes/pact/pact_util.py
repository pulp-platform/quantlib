from functools import partial
from ...util.tracing import LeafTracer, custom_symbolic_trace

from quantlib.algorithms.pact import *



#PACT operations which should not be traced through
PACT_OPS = set([PACTUnsignedAct,
            PACTAsymmetricAct,
            PACTConv2d,
            PACTConv1d,
            PACTLinear])

PACT_OPS_INT = set([PACTIntegerAdd,
                    PACTIntegerConcat,
                    PACTIntegerMatmul])

#All PACT operations - ordinarily we would want to trace through the
#integerized operations
PACT_OPS_INCLUSIVE = PACT_OPS | PACT_OPS_INT


PACTTracer = LeafTracer(leaf_types=list(PACT_OPS))
PACTInclusiveTracer = LeafTracer(leaf_types=list(PACT_OPS_INCLUSIVE))

PACT_symbolic_trace = partial(custom_symbolic_trace, tracer=PACTTracer)
PACT_symbolic_trace_inclusive = partial(custom_symbolic_trace, tracer=PACTInclusiveTracer)
