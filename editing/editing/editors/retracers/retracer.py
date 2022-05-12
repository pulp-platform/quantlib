import torch.fx as fx

from ..base import Annotator
from quantlib.editing.graphs.fx import SymbolicTraceFnType


class Retracer(Annotator):

    def __init__(self, name: str, symbolic_trace_fn: SymbolicTraceFnType):
        super(Retracer, self).__init__(name, symbolic_trace_fn)

    def apply(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:
        return self._symbolic_trace_fn(root=g)
