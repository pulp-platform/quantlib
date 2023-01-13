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

import copy
import torch
import torch.nn as nn
from typing import Tuple, Dict, Any

from quantlib.algorithms.qalgorithms import ModuleMapping
from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType

from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule


class HarmonisedAdd(nn.Module):

    def __init__(self,
                 n_inputs:                 int,
                 qgranularityspec:         QGranularitySpecType,
                 qrangespec:               QRangeSpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 mapping:                  ModuleMapping,
                 kwargs:                   Dict[str, Any],
                 use_output_scale:         bool):

        super(HarmonisedAdd, self).__init__()

        self._input_qmodules: nn.ModuleList = nn.ModuleList(list(HarmonisedAdd.get_qmodule(qgranularityspec, qrangespec, qhparamsinitstrategyspec, mapping, kwargs) for i in range(0, n_inputs)))
        self._output_qmodule: _QModule = HarmonisedAdd.get_qmodule(qgranularityspec, qrangespec, qhparamsinitstrategyspec, mapping, kwargs)

        self._use_output_scale: bool = use_output_scale

    @staticmethod
    def get_qmodule(qgranularityspec:         QGranularitySpecType,
                    qrangespec:               QRangeSpecType,
                    qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                    mapping:                  ModuleMapping,
                    kwargs:                   Dict[str, Any]) -> _QModule:

        qgranularityspec = copy.deepcopy(qgranularityspec)
        qrangespec = copy.deepcopy(qrangespec)
        qhparamsinitstrategyspec = copy.deepcopy(qhparamsinitstrategyspec)
        kwargs = copy.deepcopy(kwargs)

        qmodule_class = mapping[nn.Identity]

        qmodule = qmodule_class(qrangespec=qrangespec,
                                qgranularityspec=qgranularityspec,
                                qhparamsinitstrategyspec=qhparamsinitstrategyspec,
                                **kwargs)

        return qmodule

    # TODO:
    # All the ``_QModule``s in a ``Harmoniser`` should be kept synchronised.
    # This synchronisation involves the attributes that are not functional to
    # the harmonisation itself (e.g., ``training`` and ``_is_quantised`` flags).

    @property
    def is_training(self) -> bool:
        are_input_qmodules_training = all(map(lambda qm: qm.training, self._input_qmodules))
        is_output_qmodule_training = self._output_qmodule.training
        return are_input_qmodules_training and is_output_qmodule_training

    @property
    def is_quantised(self) -> bool:
        are_input_qmodules_quantised = all(map(lambda qm: qm._is_quantised, self._input_qmodules))
        is_output_qmodule_quantised = self._output_qmodule._is_quantised
        return are_input_qmodules_quantised and is_output_qmodule_quantised

    def start_observing(self) -> None:
        for qm in self._input_qmodules:
            qm.start_observing()
        self._output_qmodule.start_observing()

    def stop_observing(self) -> None:
        for qm in self._input_qmodules:
            qm.stop_observing()
        self._output_qmodule.stop_observing()

    def harmonise(self) -> None:

        if self._use_output_scale:
            ref_module = self._output_qmodule
        else:
            raise NotImplementedError

        scale = ref_module.scale.detach().clone()  # https://discuss.pytorch.org/t/difference-between-detach-clone-and-clone-detach/34173/2
        clip_lo = ref_module.clip_lo.detach().clone()
        clip_hi = ref_module.clip_hi.detach().clone()

        for qm in self._input_qmodules:
            qm.scale.data.copy_(scale)
            qm.clip_lo.data.copy_(clip_lo)
            qm.clip_hi.data.copy_(clip_hi)

    def forward(self, *args: Tuple[torch.Tensor]) -> torch.Tensor:

        if self.is_training and self.is_quantised:  # TODO: this should happen also during the validation, but if we do then `torch.fx` `Tracer`s also track all the operations in the `harmonise` method!
            self.harmonise()

        sum_ = self._input_qmodules[0](args[0])
        for i, (qm, x) in enumerate(zip(self._input_qmodules[1:], args[1:])):
            sum_ = sum_ + qm(x)

        return self._output_qmodule(sum_)
