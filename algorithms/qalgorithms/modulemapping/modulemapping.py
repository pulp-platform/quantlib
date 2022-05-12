from collections import OrderedDict
import torch.nn as nn
from typing import Type

from ...qmodules.qmodules.qmodules import _QModule
from ...qmodules import SUPPORTED_FPMODULES


class ModuleMapping(OrderedDict):
    """Map floating-point ``nn.Module``s to their fake-quantised counterparts.

    QuantLib developers who implement new PTQ/QAT algorithms should define a
    ``ModuleMapping`` object inside each algorithm-specific sub-package, then
    add it to the global register of PTQ/QAT algorithms.

    """

    def __setitem__(self, fpmodule: Type[nn.Module], fqmodule: Type[_QModule]):

        if not isinstance(fpmodule, type(nn.Module)):
            raise TypeError  # not a floating-point module
        if not isinstance(fqmodule, type(_QModule)):
            raise TypeError  # not a fake-quantised module

        if not (fpmodule in SUPPORTED_FPMODULES):
            raise ValueError  # QuantLib does not support a fake-quantised counterpart for this `nn.Module` class

        super(ModuleMapping, self).__setitem__(fpmodule, fqmodule)
