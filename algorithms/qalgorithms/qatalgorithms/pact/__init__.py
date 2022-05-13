from quantlib.algorithms.qalgorithms import ModuleMapping

from .qactivations import NNMODULE_TO_PACTACTIVATION, PACTIdentity, PACTReLU, PACTReLU6, PACTLeakyReLU
from .qlinears     import NNMODULE_TO_PACTLINEAR, PACTLinear, PACTConv1d, PACTConv2d, PACTConv3d

NNMODULE_TO_PACTMODULE = ModuleMapping(**NNMODULE_TO_PACTACTIVATION, **NNMODULE_TO_PACTLINEAR)

from .optimisers import PACTSGD, PACTAdam, PACTAdagrad
