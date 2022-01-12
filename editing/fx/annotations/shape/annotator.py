import collections
import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp

from typing import Dict, Union


InputSpecification = collections.namedtuple('InputDescription', ['shape', 'dtype'])


class ShapeAnnotator(object):

    def __init__(self, gm: fx.GraphModule):
        self._gm = gm

    @property
    def gm(self) -> fx.GraphModule:
        return self._gm

    @staticmethod
    def is_shape_annotated(n: fx.Node):
        return 'tensor_meta' in n.meta.keys()

    def apply(self, inputs_spec: Dict[str, Union[torch.Size, InputSpecification]]):

        # canonicalise input
        for k, v in inputs_spec.items():
            if not isinstance(v, tuple):
                inputs_spec[k] = InputSpecification(shape=v, dtype=torch.float32)  # default data type for input `torch.Tensor`s is `torch.float32`

        inputs = [torch.ones(s, dtype=d) for s, d in inputs_spec.values()]
        ShapeProp(self.gm).run(*inputs)
