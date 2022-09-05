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
import numpy as np
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)

_QTENSOR_OVERRIDES = {}

def qt_implements(torch_func : callable):
    def inner(f):
        _QTENSOR_OVERRIDES[torch_func] = f
        return f
    return inner


@qt_implements(torch.stack)
def qt_stack(tensors, dim=0, out=None):
    with torch._C.DisableTorchFunction():
        stacked = torch.stack(tensors, dim=dim, out=out)

    if all(isinstance(t, QTensor) and t.eps is not None for t in tensors):
        epses = [t.eps for t in tensors]
        eps_diffs = [np.abs(e1 - e2) for e1, e2 in zip(epses[:-1], epses[1:])]
        if not all(ed < 1e-8 for ed in eps_diffs):
            print("Warning: stacking QTensors  with different eps values!! Eps is discarded for resulting QTensor")
        else:
            stacked.eps = epses[0]
    return stacked



class QTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, x, eps=None, *args, **kwargs):

        if isinstance(x, torch.Tensor):
            inst = x.__deepcopy__(memo={}).as_subclass(cls)
        else:
            inst = super().__new__(cls, x, *args, **kwargs)

        return inst

    def __init__(self, x, eps=None, *args, **kwargs):
        if eps is not None:
            if isinstance(x, torch.Tensor):
                self._eps = torch.as_tensor(eps).type_as(x)
            else:
                self._eps = torch.as_tensor(eps)
                self._eps.requires_grad = False

    @property
    def eps(self):
        if hasattr(self, '_eps'):
            return self._eps
        else:
            return None

    @eps.setter
    def eps(self, value):
        self._eps = value

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
        if kwargs is None:
            kwargs = {}
        if func.__name__ in cls.getOverriddenMethods():
            return getattr(cls, func.__name__)(*args, **kwargs)
        elif func in _QTENSOR_OVERRIDES:
            return _QTENSOR_OVERRIDES[func](*args, **kwargs)
        else:
            ret = super().__torch_function__(func,types,args,kwargs)
            c = _convert(ret, cls)
        return c

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
        #import ipdb; ipdb.set_trace()
        with torch._C.DisableTorchFunction():
            tempTensor = super().to(*args, **kwargs)
        new_obj.data=tempTensor.data
        new_obj.requires_grad=tempTensor.requires_grad
        if hasattr(self, '_eps'):
            new_obj.__init__(tempTensor, self._eps)
        else:
            new_obj.__init__(tempTensor, None)
        return(new_obj)

    def split(self, *args, **kwargs):
        with torch._C.DisableTorchFunction():
            base_spl = super().split(*args, **kwargs)
        q_spl = _convert(base_spl, QTensor)
        if self.eps is not None:
            for qt in q_spl:
                qt.eps = self.eps
        return q_spl




def _convert(ret, cls):
    if isinstance(ret, torch.Tensor) and not isinstance(ret, cls):
        ret = ret.as_subclass(cls)
    if isinstance(ret, (tuple, list)):
        # Also handles things like namedtuples
        ret = type(ret)(_convert(r, cls) for r in ret)

    return ret
