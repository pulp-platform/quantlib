# ----------------------------------------------------------------------
#
# File: QTensor.py
#
# Last edited: 06.01.2022        
# 
# Copyright (C) 2022, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function, get_default_nowrap_functions)

class QTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, eps=None, *args, **kwargs):
        if isinstance(x, torch.Tensor):
            return x.__deepcopy__(memo={}).as_subclass(cls)
        else:
            return super().__new__(cls, x, *args, **kwargs)

    def __init__(self, x, eps=None, **kwargs):
        if eps is not None:
            if isinstance(x, torch.Tensor):
                self._eps = torch.as_tensor(eps).type_as(x)
            else:
                self._eps = torch.as_tensor(eps)
            self._eps.requires_grad = False
        else:
            self._eps = None

    @classmethod
    def getOverriddenMethods(cls):
        parent_attrs = set()
        for base in cls.__bases__:
            parent_attrs.update(dir(base))

        # find all methods implemented in the class itself
        methods = {name for name, thing in vars(cls).items() if callable(thing)}

        # return the intersection of both
        return parent_attrs.intersection(methods)
        
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        return super().__torch_function__(func,types,args,kwargs)
        
    def clone(self, *args, **kwargs):
        if hasattr(self, '_eps'):
            return QTensor(super().clone(*args, **kwargs), self._eps)
        else:
            return QTensor(super().clone(*args, **kwargs), None)
        
    def to(self, *args, **kwargs):
        if hasattr(self, '_eps'):
            new_obj=QTensor([], self._eps)
        else:
            new_obj=QTensor([], None)
        tempTensor=super().to(*args, **kwargs)
        new_obj.data=tempTensor.data
        new_obj.requires_grad=tempTensor.requires_grad
        if hasattr(self, '_eps'):
            new_obj.__init__(tempTensor, self._eps)
        else:
            new_obj.__init__(tempTensor, None)
        return(new_obj)            

    
