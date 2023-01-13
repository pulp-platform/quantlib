# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

from typing import Callable
import quantlib.algorithms as qa
from quantlib.editing.editing.float2fake.quantisation.activationrounder import ActivationRounder
from quantlib.editing.editing.float2fake.quantisation.weightrounder import WeightRounder
from quantlib.editing.editing.float2fake.quantisation.addcalibrator import AddCalibrator
from . import canonicalisation
from . import quantisation
from contextlib import contextmanager
import torch.nn as nn

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
        qinterposerqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True}, 'minmax', 'PACT')

        super(F2F8bitPACTConverter, self).__init__(
            modulewisedescriptionspec,
            addtreeqdescriptionspec,
            addtreeforceoutputeps,
            qinterposerqdescriptionspec
        )

class F2F8bitPACTRoundingConverter(ComposedEditor):
    """Standard float-to-fake converter mapping all floating-point
    ``nn.Module``s to 8-bit quantised counterparts, without requiring
    QAT.

    The weight parameters of linear operations are mapped to signed 8-bit,
    whereas features are mapped to unsigned 8-bit. The algorithm is PACT.
    This converter initializes quantization parameters so that they work
    reasonably (with MobileNetV1, V2) without any QAT, in combination
    with rounding of weights and activations (F2F8bitPACTRounder, to be
    applied *after* the quantization hyperparameters have been calibrated).

    """
    def __init__(self):

        # `ModuleWiseConverter` argument
        modulewisedescriptionspec = (
            ({'types': ('ReLU',)},                                 ('per-array',              {'bitwidth': 8, 'signed': False}, 'minmax',                        'PACT')),
            ({'types': ('ReLU6',)},                                ('per-array',              {'bitwidth': 8, 'signed': False}, ('const', {'a': 0.0, 'b': 6.0}), 'PACT')),
            ({'types': ('Identity',)},                             ('per-array',              {'bitwidth': 8, 'signed': True},  'const',                         'PACT')),  # using an unsigned data type for the identity would clamp all negative inputs to zero
            ({'types': ('Linear', 'Conv1d', 'Conv2d', 'Conv3d',)}, ('per-outchannel_weights', {'bitwidth': 8, 'signed': True},  'minmax',                        'PACT')),
        )

        # `AddTreeHarmoniser` argument
        addtreeqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True}, ('meanstd', {'n_std': 5}), 'PACT')
        addtreeforceoutputeps = True

        # `QuantiserInterposer` argument
        qinterposerqdescriptionspec = ('per-array', {'bitwidth': 8, 'signed': True}, 'minmax', 'PACT')

        super(F2F8bitPACTRoundingConverter, self).__init__([
            F2FCanonicaliser(),
            F2FQuantiser(
                modulewisedescriptionspec,
                addtreeqdescriptionspec,
                addtreeforceoutputeps,
                qinterposerqdescriptionspec
            )
        ])

class F2F8bitPACTRounder(ComposedEditor):
    """Editor adding rounding to all PACT modules (Linear and Act).

    This rounder should be used after the network has been quantized (e.g.,
    with `F2F8bitPACTRoundingConverter`) and calibrated using 
    ```
    with qe.float2fake.calibration(net):
        validate(net)
    ```
    Besides rounding, it also harmonizes add nodes.
    FIXME: maybe change the name to Calibrator or similar.

    """
    def __init__(self):

        super(F2F8bitPACTRounder, self).__init__([
            WeightRounder(),
            ActivationRounder(),
            AddCalibrator()
        ])

@contextmanager
def calibration(net : nn.Module, verbose : bool = False):
    """Statistics-based calibration context.

    """

    if verbose:
        print("[Entering calibration mode]")
    for m in net.modules():
        if isinstance(m, tuple(qa.qalgorithms.qatalgorithms.pact.NNMODULE_TO_PACTMODULE.values())):
            m.start_observing()
    try:
        yield
    finally:
        if verbose:
            print("[Exiting calibration mode]")
        for m in net.modules():
            if isinstance(m, tuple(qa.qalgorithms.qatalgorithms.pact.NNMODULE_TO_PACTMODULE.values())):
                m.stop_observing()
