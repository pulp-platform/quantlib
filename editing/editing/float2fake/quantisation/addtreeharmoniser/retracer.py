import torch.fx as fx

from quantlib.editing.editing.editors import Annotator
from .tracing import quantlib_harmonisedadd_symbolic_trace


class QuantLibHarmonisedAddRetracer(Annotator):

    def __init__(self):
        super(QuantLibHarmonisedAddRetracer, self).__init__('QuantLibHarmonisedAddTracer', quantlib_harmonisedadd_symbolic_trace)

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        return self._symbolic_trace_fn(root=g)
