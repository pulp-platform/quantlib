from . import canonicalisation
from . import quantisation

#
# In the following, we define high-level `Editor`s (i.e., `ComposedEditor`s)
# to transform floating-point PyTorch networks into fake-quantised ones.
#
# Under the hood, `F2FConverter` breaks down into 17 base `Rewriter`s:
# * `ReLUModulariser`;
# * `ReLU6Modulariser`;
# * `LeakyReLUModulariser`;
# * `LinearBN1dBiasFolder`;
# * `Conv1dBN1dBiasFolder`;
# * `Conv2dBN2dBiasFolder`;
# * `Conv3dBN3dBiasFolder`;
# * `ModuleWiseConverter`;
# * `AddTreeHarmoniser`;
# * `LinearLinearQuantiserInterposer`;
# * `Conv1dConv1dQuantiserInterposer`;
# * `Conv2dConv2dQuantiserInterposer`;
# * `Conv3dConv3dQuantiserInterposer`;
# * `BN1dLinearQuantiserInterposer`;
# * `BN1dConv1dQuantiserInterposer`;
# * `BN2dConv2dQuantiserInterposer`;
# * `BN3dConv3dQuantiserInterposer`.
#
# We also implement an example `F2F8bitPACTConverter` showing how to inherit
# from the general-purpose `F2FConverter` to derive precision- and algorithm-
# specific F2F conversion flows.
#

from .canonicalisation import F2FCanonicaliser
from .quantisation import F2FQuantiser
from .quantisation.modulewiseconverter.modulewisedescription import ModuleWiseDescriptionSpecType
from .quantisation.qdescription import QDescriptionSpecType
from quantlib.editing.editing.editors import ComposedEditor


class F2FConverter(ComposedEditor):
    """General-purpose converter to map floating-point networks into
    fake-quantised ones.
    """
    def __init__(self,
                 modulewisedescriptionspec:   ModuleWiseDescriptionSpecType,
                 addtreeqdescriptionspec:     QDescriptionSpecType,
                 addtreeforceoutputeps:       bool,
                 qinterposerqdescriptionspec: QDescriptionSpecType):

        super(F2FConverter, self).__init__([
            F2FCanonicaliser(),
            F2FQuantiser(
                modulewisedescriptionspec,
                addtreeqdescriptionspec,
                addtreeforceoutputeps,
                qinterposerqdescriptionspec
            ),
        ])


class F2F8bitPACTConverter(F2FConverter):
    """Standard float-to-fake converter mapping all floating-point
    ``nn.Module``s to 8-bit quantised counterparts.

    The weight parameters of linear operations are mapped to signed 8-bit,
    whereas features are mapped to unsigned 8-bit. The QAT algorithm is PACT.

    """
    def __init__(self):

        # `ModuleWiseConverter` argument
        modulewisedescriptionspec = (
            ({'types': ('ReLU',   'ReLU6',  'LeakyReLU',)},        ('per-array',              {'bitwidth': 8, 'signed': False}, ('const', {'a': 0.0, 'b': 6.0}), 'PACT')),
            ({'types': ('Identity',)},                             ('per-array',              {'bitwidth': 8, 'signed': True},  'const',                         'PACT')),  # using an unsigned data type for the identity would clamp all negative inputs to zero
            ({'types': ('Linear', 'Conv1d', 'Conv2d', 'Conv3d',)}, ('per-outchannel_weights', {'bitwidth': 8, 'signed': True},  'minmax',                        'PACT')),
        )

        # `AddTreeHarmoniser` argument
        addtreeqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True}, 'minmax', 'PACT')
        addtreeforceoutputeps = True

        # `QuantiserInterposer` argument
        qinterposerqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True},  'minmax', 'PACT')

        super(F2F8bitPACTConverter, self).__init__(
            modulewisedescriptionspec,
            addtreeqdescriptionspec,
            addtreeforceoutputeps,
            qinterposerqdescriptionspec
        )
