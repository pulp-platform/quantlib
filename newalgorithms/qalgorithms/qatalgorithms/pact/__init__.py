from .qactivations import NNMODULE_TO_PACTACTIVATION
from .qlinears     import NNMODULE_TO_PACTLINEAR

NNMODULE_TO_PACTMODULE = {**NNMODULE_TO_PACTACTIVATION, **NNMODULE_TO_PACTLINEAR}

from .qactivations import PACTIdentity, PACTReLU, PACTReLU6, PACTLeakyReLU
from .qlinears     import PACTLinear, PACTConv1d, PACTConv2d, PACTConv3d
