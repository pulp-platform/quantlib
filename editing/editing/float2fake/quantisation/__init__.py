from .qdescription import QDescription
from .modulewiseconverter import *
from .addtreeharmoniser import *
from .quantiserinterposer import *
from .activationrounder import *
from .weightrounder import *
from .addcalibrator import *

#
# In the following, we define a high-level `Editor` (i.e., a `ComposedEditor`)
# to transform canonical floating-point QuantLib networks into fake-quantised
# ones.
#
# `F2FQuantiser` breaks down into three `Rewriter`s:
# * `ModuleWiseConverter`, mapping selected floating-point `nn.Module`s in the
#   input floating-point network to fake-quantised counterparts;
# * `AddTreeHarmoniser`, ensuring that additions between fake-quantised
#   `torch.Tensor`s can be mapped to integer additions during fake-to-true
#   conversion;
# * `QuantiserInterposer`, ensuring that we do not chain fake-quantised linear
#   operations.
#

from .qdescription import QDescriptionSpecType
from .modulewiseconverter.modulewisedescription import ModuleWiseDescriptionSpecType
from quantlib.editing.editing.editors import ComposedEditor
from quantlib.editing.editing.editors.retracers import QuantLibRetracer


class F2FQuantiser(ComposedEditor):
    """General-purpose ``Rewriter`` mapping floating-point PyTorch networks to
    fake-quantised ones.
    """
    def __init__(self,
                 modulewisedescriptionspec:   ModuleWiseDescriptionSpecType,
                 addtreeqdescriptionspec:     QDescriptionSpecType,
                 addtreeforceoutputeps:       bool,
                 qinterposerqdescriptionspec: QDescriptionSpecType):

        super(F2FQuantiser, self).__init__([
            QuantLibRetracer(),
            ModuleWiseConverter(modulewisedescriptionspec),
            QuantLibHarmonisedAddRetracer(),
            AddTreeHarmoniser(
                addtreeqdescriptionspec,
                addtreeforceoutputeps
            ),
            # QuantLibRetracer(),  # TODO: if we retrace now, calls to `gm.graph.recompile()` executed downstream will remove the control structure defined by the `forward` method of `HarmonisedAdd`, since these container `nn.Module`s have become "invisible"
            QuantiserInterposer(
                qinterposerqdescriptionspec
            ),
        ])
