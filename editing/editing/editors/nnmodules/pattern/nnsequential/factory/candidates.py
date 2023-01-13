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

from collections import OrderedDict
import torch.nn as nn
from typing import NamedTuple, Tuple, Dict, Union, Optional, Type, Any

from ...base import Checker


class NNModuleDescription(NamedTuple):
    class_: Type[nn.Module]
    kwargs: Dict[str, Any]
    checkers: Union[Checker, Tuple[Checker, ...]] = tuple()


CandidateName = str
CandidateDescription = Union[NNModuleDescription, None]


class Candidates(OrderedDict):

    def __setitem__(self, name: CandidateName, candidate: CandidateDescription):

        # validate input types
        if not isinstance(name, str):
            raise TypeError
        if not (isinstance(candidate, NNModuleDescription) or (candidate is None)):
            raise TypeError

        super(Candidates, self).__setitem__(name, candidate)
