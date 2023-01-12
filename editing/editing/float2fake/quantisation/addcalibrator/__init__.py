import quantlib.editing.editing as qe
import quantlib.editing.graphs as qg
from quantlib.editing.editing.editors import Annotator
from quantlib.editing.graphs.fx import quantlib_symbolic_trace
import torch.nn as nn
import torch.fx as fx

# harmonises the HarmonisedAdd post-calibration
class AddCalibrator(Annotator):

    def __init__(self):
        super(AddCalibrator, self).__init__('AddCalibrator', quantlib_symbolic_trace)

    def apply(self,
              g: fx.GraphModule):
        for _, m in g.named_modules():
            if m.__class__.__name__ == "HarmonisedAdd":
                m.harmonise()
        return g

__all__ = [
    'AddCalibrator',
]
