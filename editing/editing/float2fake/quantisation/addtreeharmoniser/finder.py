import operator
import torch

from quantlib.editing.editing.editors.optrees import OpSpec, OpTreeFinder
from quantlib.editing.graphs.fx import FXOpcodeClasses


addspec = OpSpec([
    (next(iter(FXOpcodeClasses.CALL_FUNCTION.value)), (operator.add, torch.add,)),
    (next(iter(FXOpcodeClasses.CALL_METHOD.value)),   ('add',)),
])


class AddTreeFinder(OpTreeFinder):
    def __init__(self):
        super(AddTreeFinder, self).__init__(addspec)
