#
# pact_optimizers.py
#
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
#
# Copyright (c) 2020-2022 ETH Zurich.
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

import torch.nn as nn
import torch.optim
from typing import Type

from . import NNMODULE_TO_PACTMODULE


PACTMODULE_CLASSES = tuple(class_ for class_ in NNMODULE_TO_PACTMODULE.values())


class PACTOptimiserFactory(object):  # https://stackoverflow.com/questions/21060073/dynamic-inheritance-in-python

    def get_pact_optimiser_class(self, base_optimiser_class: Type[torch.optim.Optimizer]) -> Type[torch.optim.Optimizer]:

        class_name = "PACT" + base_optimiser_class.__name__

        def __init__(self_, network: nn.Module, pact_decay: float, *args, **kwargs):
            # `args`: positional arguments for the base optimiser class
            # `kwargs`: keyword arguments for the base optimiser class

            pact_modules = list(filter(lambda m: isinstance(m, PACTMODULE_CLASSES), network.modules()))

            # split PACT-learnable parameters from the remaining ones
            pact_learnable_clipping_params = [p for m in pact_modules for p in (m.clip_lo, m.clip_hi) if p.requires_grad]
            other_params = [p_ for p_ in network.parameters() if not any(p_ is p for p in pact_learnable_clipping_params)]

            # Note: when the learnable clipping parameters are not used
            # (e.g., while quantization is disabled), the weight decay
            # will have NO effect on them. This fact is very handy,
            # since it prevents us from having to update their
            # 'requires_grad' flags when turning quantization on/off.
            base_optimiser_class.__init__(self_,
                                          ({'params':       pact_learnable_clipping_params,
                                            'weight_decay': pact_decay},
                                           {'params':       other_params}),
                                          *args,
                                          **kwargs)

        pact_optimizer_class = type(class_name, (base_optimiser_class,), {'__init__': __init__})

        return pact_optimizer_class


factory = PACTOptimiserFactory()

PACTSGD     = factory.get_pact_optimiser_class(torch.optim.SGD)
PACTAdam    = factory.get_pact_optimiser_class(torch.optim.Adam)
PACTAdagrad = factory.get_pact_optimiser_class(torch.optim.Adagrad)
