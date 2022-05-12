import torch.fx as fx

from .baseeditor import BaseEditor, SymbolicTraceFnType


class Annotator(BaseEditor):
    """Base ``Editor`` representing an annotation.

    Its application does not change the topology of the graph, but only its
    attributes (i.e., it modifies or enriches the semantics of the graph).

    """

    def __init__(self,
                 name:              str,
                 symbolic_trace_fn: SymbolicTraceFnType):
        super(Annotator, self).__init__(name, symbolic_trace_fn)

    def apply(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:
        raise NotImplementedError
