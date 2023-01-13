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

from .activationmodulariser import *
from .linearopbnbiasfolder import *
from .flattencanonicaliser import *

#
# In the following, we define a high-level `Editor` (i.e., a `ComposedEditor`)
# to map floating-point PyTorch networks into a canonical form.
#
# Canonical floating-point QuantLib networks satisfy the following properties:
# * they implement activations using only calls to the modular API (i.e.,
#   `nn.Module` objects);
# * those linear operations that are followed by batch-normalisation ones will
#   have their bias absorbed into the batch-normalisation's mean.
#
# Under the hood, `F2FCanonicaliser` breaks down into seven base `Rewriter`s:
# * `ReLUModulariser`;
# * `ReLU6Modulariser`;
# * `LeakyReLUModulariser`;
# * `LinearBN1dBiasFolder`;
# * `Conv1dBN1dBiasFolder`;
# * `Conv2dBN2dBiasFolder`;
# * `Conv3dBN3dBiasFolder`.
#

from quantlib.editing.editing.editors import ComposedEditor
from quantlib.editing.editing.editors.retracers import QuantLibRetracer


class F2FCanonicaliser(ComposedEditor):
    """General-purpose ``Rewriter`` mapping PyTorch floating-point networks
    into canonical QuantLib floating-point networks.
    """
    def __init__(self):
        super(F2FCanonicaliser, self).__init__([
            QuantLibRetracer(),
            ActivationModulariser(),
            QuantLibRetracer(),
            LinearOpBNBiasFolder(),
            FlattenCanonicaliser()
        ])
