from collections import OrderedDict
from typing import Dict

from .utils import ModuleMapping

from quantlib.algorithms.qalgorithms.qatalgorithms.pact import NNMODULE_TO_PACTMODULE
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import PACTIdentity, PACTReLU, PACTReLU6, PACTLeakyReLU
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import PACTLinear, PACTConv1d, PACTConv2d, PACTConv3d


class PTQQATIndex(object):
    """This object is supposed to be unique (i.e., a singleton)."""
    def __init__(self):
        object.__init__(self)
        self._register: Dict[str, ModuleMapping] = OrderedDict()

    @property
    def register(self) -> Dict[str, ModuleMapping]:
        return self._register

    def register_algorithm(self,
                           name: str,
                           fp_to_fq: ModuleMapping):

        name = name.upper()
        self._register[name] = fp_to_fq


# create indices
ptqqat_index = PTQQATIndex()
# register PACT
ptqqat_index.register_algorithm('PACT', NNMODULE_TO_PACTMODULE)
