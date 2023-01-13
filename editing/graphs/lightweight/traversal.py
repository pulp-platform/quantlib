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

from functools import partial
import torch.nn as nn
from typing import Type, Tuple

from .node import LightweightNode, LightweightNodeList
from ..nn import EpsTunnel, Requantisation
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule


def custom_traverse(root:           nn.Module,
                    qualified_name: str = '',
                    leaf_types:     Tuple[Type[nn.Module], ...] = tuple()) -> LightweightNodeList:
    """Traverse the ``nn.Module`` tree rooted at ``root`` down to
    user-specified atoms.

    In some scenarios, users might want to identify container ``nn.Module``s
    as opposed to atomic ``nn.Module``s. This function traverses containers
    unless their type matches one of the user-specified ones.

    """

    nodes = LightweightNodeList()

    if isinstance(root, leaf_types) or len(list(root.named_children())) == 0:
        nodes.append(LightweightNode(name=qualified_name, module=root))

    else:
        for child_name, child_module in root.named_children():
            child_qualified_name = child_name if qualified_name == '' else '.'.join([qualified_name, child_name])
            nodes.extend(custom_traverse(root=child_module, qualified_name=child_qualified_name, leaf_types=leaf_types))

    return nodes


quantlib_traverse = partial(custom_traverse, leaf_types=(_QModule, EpsTunnel, Requantisation))
