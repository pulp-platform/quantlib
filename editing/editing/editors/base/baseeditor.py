import torch.fx as fx

from .editor import Editor
from ....graphs.fx import SymbolicTraceFnType


class BaseEditor(Editor):

    def __init__(self,
                 name:              str,
                 symbolic_trace_fn: SymbolicTraceFnType):

        super(BaseEditor, self).__init__()

        self._id: str = '_'.join(['QL', name + f'_{str(id(self))}_'])  # we use this attribute to uniquely identify the edits made using this `Editor`
        self._symbolic_trace_fn = symbolic_trace_fn                    # we assume that the `fx.GraphModule`s processed by this `Editor` have been obtained using this tracing function

    @property
    def id_(self) -> str:
        return self._id

    def apply(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:
        raise NotImplementedError
