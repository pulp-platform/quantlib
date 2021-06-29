# 
# nodes.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
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

import re
import torch
from enum import IntEnum, unique


__NODE_ID_FORMAT__ = '{:06d}'


@unique
class Bipartite(IntEnum):
    KERNEL = 0
    MEMORY = 1
    CONTXT = 2


@unique
class DataPartition(IntEnum):
    INPUT     = 0
    OUTPUT    = 1
    PARAMETER = 2
    OTHER     = 3


class QuantLabNode(object):

    def __init__(self, obj):
        self.nobj = obj


class ONNXNode(QuantLabNode):

    def __init__(self, obj):
        super(ONNXNode, self).__init__(obj)

    @staticmethod
    def onnx_scope_2_pytorch_scope(onnx_scope):
        module_name_parts = re.findall('\[.*?\]', onnx_scope)
        pytorch_scope = '.'.join([mn[1:-1] for mn in module_name_parts])
        return pytorch_scope

    @property
    def ntype(self):
        if isinstance(self.nobj, torch._C.Node):
            ntype = self.nobj.kind()
        elif isinstance(self.nobj, torch._C.Value):
            ntype = '*'  # data nodes are untyped ('onnx::Tensor'?)
        return ntype

    @property
    def nscope(self):
        if isinstance(self.nobj, torch._C.Node):
            nscope = ONNXNode.onnx_scope_2_pytorch_scope(self.nobj.scopeName())
        elif isinstance(self.nobj, torch._C.Value):
            nscope = self.nobj.debugName()
        return nscope


class PyTorchNode(QuantLabNode):

    def __init__(self, obj):
        super(PyTorchNode, self).__init__(obj)

    @property
    def ntype(self):
        if isinstance(self.nobj, torch.nn.Module):
            ntype = self.nobj.__class__.__name__
        elif isinstance(self.nobj, torch._C.Value):
            ntype = '*'  # data nodes are untyped ('onnx::Tensor'?)
        return ntype

    @property
    def nscope(self):
        if isinstance(self.nobj, torch.nn.Module):
            nscope = ''  # the scope of `nn.Module`s usually depends on the "view" that the network's coder had of it at implementation time; we leave op nodes unscoped
        elif isinstance(self.nobj, torch._C.Value):
            nscope = self.nobj.debugName()
        return nscope

