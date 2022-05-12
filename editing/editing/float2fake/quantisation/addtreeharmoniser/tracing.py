from functools import partial

from quantlib.editing.graphs.fx import QuantLibTracer, custom_symbolic_trace
from quantlib.editing.graphs.nn import HarmonisedAdd


class QuantLibHarmonisedAddTracer(QuantLibTracer):
    def __init__(self):
        other_leaf_types = (HarmonisedAdd,)
        super(QuantLibHarmonisedAddTracer, self).__init__(other_leaf_types)


quantlib_harmonisedadd_symbolic_trace = partial(custom_symbolic_trace, tracer=QuantLibHarmonisedAddTracer())
