from collections import OrderedDict

from .modulemapping import ModuleMapping
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import NNMODULE_TO_PACTMODULE
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import PACTIdentity, PACTReLU, PACTReLU6, PACTLeakyReLU
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import PACTLinear, PACTConv1d, PACTConv2d, PACTConv3d


class PTQQATRegister(OrderedDict):
    """This object is supposed to be unique (i.e., a singleton)."""

    def __setitem__(self,
                    acronym:  str,
                    fp_to_fq: ModuleMapping):  # floating-point to fake-quantised

        # validate input type
        if not isinstance(acronym, str):
            raise TypeError  # the acronym for a PTQ/QAT algorithm should be a string
        if not isinstance(fp_to_fq, ModuleMapping):
            raise TypeError

        # canonicalise
        acronym = acronym.upper()

        # validate input value
        if acronym in self.keys():
            raise ValueError  # the acronym has already be used for another PTQ/QAT algorithm

        super(PTQQATRegister, self).__setitem__(acronym, fp_to_fq)


# create indices
register = PTQQATRegister()

# -- TO BE EXPANDED BY IMPLEMENTERS OF PTQ/QAT ALGORITHMS -- #
# register PACT
register['PACT'] = NNMODULE_TO_PACTMODULE
