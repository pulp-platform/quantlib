import torch
import torch.fx as fx
from torch.fx.passes.shape_prop import ShapeProp
import warnings
from typing import Dict, NamedTuple, Union

from ...editors.editors import Annotator
from ....graphs import FXOPCODE_PLACEHOLDER
from quantlib.utils import quantlib_err_header, quantlib_wng_header


def is_shape_annotated(n: fx.Node):
    return 'tensor_meta' in n.meta.keys()


class InputSpecification(NamedTuple):
    shape: torch.Size
    dtype: torch.dtype


DEFAULT_DTYPE = torch.float32


class ShapePropagator(Annotator):

    def __init__(self):
        name = 'ShapePropagator'
        super(ShapePropagator, self).__init__(name)

    @staticmethod
    def clear_shape(g: fx.GraphModule):
        """Remove all shape annotations from the given graph."""
        for n in g.graph.nodes:
            try:
                del n.meta['tensor_meta']
            except KeyError:
                pass

    def apply(self, g: fx.GraphModule, input_specs: Dict[str, Union[InputSpecification, torch.Size]] = {}) -> fx.GraphModule:

        # verify that each input is specified
        inputs_specified   = set(input_specs.keys())
        inputs_graph       = {n.name for n in g.graph.nodes if n.op in FXOPCODE_PLACEHOLDER}
        inputs_unspecified = inputs_graph.difference(inputs_specified)
        inputs_unknown     = inputs_specified.difference(inputs_graph)
        if len(inputs_unspecified) > 0:  # I do not have all the informatio I need
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + f"requires descriptions for all input nodes, but the following were not specified: {inputs_unspecified}.")
        elif len(inputs_unknown) > 0:    # I have all the information I need, but also spurious one
            warnings.warn(quantlib_wng_header(obj_name=self.__class__.__name__) + f"received descriptions for unknown placeholder nodes: {inputs_unknown}.")
        else:                            # everything seems OK
            pass

        # canonicalise input
        for name, spec in input_specs.items():
            if isinstance(spec, torch.Size):
                input_specs[name] = InputSpecification(shape=spec, dtype=DEFAULT_DTYPE)

        inputs = [torch.ones(shape, dtype=dtype) for shape, dtype in input_specs.values()]
        ShapeProp(g).run(*inputs)

        return g
