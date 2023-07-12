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




class QTensor(torch.Tensor):

    hookedMethods = {}

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
                with torch._C.DisableTorchFunction():
                    self._eps = torch.as_tensor(eps).type_as(x)
            else:
                self._eps = torch.as_tensor(eps)
                self._eps.requires_grad = False


    def new_empty(self, *args, **kwargs):
        return super().new_empty(*args, **kwargs).as_subclass(QTensor)

    @property
    def eps(self):
        if hasattr(self, '_eps'):
            return self._eps
        else:
            return None

    @eps.setter
    def eps(self, value):
        if value is None:
            if hasattr(self, '_eps'):
                del self._eps
            return
        if isinstance(value, torch.Tensor):
            self._eps = value.clone().detach()
        else:
            self._eps = torch.as_tensor(value)

    @classmethod
    def hookMethod(cls, methodName, function):
        if methodName in cls.getOverriddenMethods():
            raise KeyError(f"Trying to override method {methodName} of QTensor!")
        cls.hookedMethods[methodName] = function

    @classmethod
    def getOverriddenMethods(cls):
        # find all methods implemented in the class itself
        methods = [name for name, thing in vars(cls).items() if callable(thing)]
        hookedMethodList = list(cls.hookedMethods.keys())
        return methods+hookedMethodList

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func.__name__ in cls.getOverriddenMethods():
            if func.__name__ not in cls.hookedMethods.keys():
                return getattr(cls, func.__name__)(*args, **kwargs)
            else:
                return cls.hookedMethods[func.__name__](*args, **kwargs)


        else:
            #print(f"not dispatching {func.__name__}")
            # by default, gather all QTensor arguments and check their
            # epsilons. If they match, set the output's epsilon to the matching
            # epsilon. Otherwise, warn the user and discard the epsilon.
            all_qt_arguments = [a for a in list(args) + list(kwargs.values()) if isinstance(a, cls)]
            all_have_eps = all(a.eps is not None for a in all_qt_arguments)
            none_have_eps = all(a.eps is None for a in all_qt_arguments)
            eps_out = None
            if not none_have_eps and not all_have_eps:
                print(f"Warning: got multiple QTensor inputs to function {func.__name__} and only some of them have an epsilon. Discarding epsilon!")
            elif all_have_eps:
                epses = [t.eps for t in all_qt_arguments]
                eps_diffs = [(e1 - e2).abs().sum() for e1, e2 in zip(epses[:-1], epses[1:])]
                if not all(ed < 1e-8 for ed in eps_diffs):
                    print(f"Warning: Called function {func.__name__} on QTensors with different eps values!! Eps is discarded for resulting QTensor.")
                else:
                    eps_out = epses[0]
            with torch._C.DisableTorchFunction():
                tens_types = tuple(torch.Tensor if t is QTensor else t for t in types)
                tens_args = tuple(torch.Tensor.as_subclass(a, torch.Tensor)  if isinstance(a, QTensor) else a for a in args)
                ret = super().__torch_function__(func,tens_types,tens_args,kwargs)
            c = _convert(ret, cls, eps=eps_out)
        return c

    def clone(self, *args, **kwargs):
        with torch._C.DisableTorchFunction():
            if hasattr(self, '_eps'):
                return QTensor(super().clone(*args, **kwargs), self._eps)
            else:
                return QTensor(super().clone(*args, **kwargs), None)

    def to(self, *args, **kwargs):
        with torch._C.DisableTorchFunction():
            new_obj = super().to(*args, **kwargs).as_subclass(QTensor)
            new_obj.eps = self.eps

        return(new_obj)

    def split(self, *args, **kwargs):
        eps_out = self.eps
        with torch._C.DisableTorchFunction():
            base_spl = super().split(*args, **kwargs)
        q_spl = _convert(base_spl, QTensor, eps=eps_out)
        return q_spl

def _convert(ret, cls, eps=None):

    if isinstance(ret, torch.Tensor) and not isinstance(ret, cls):
        #GEORGR: is this right?
        ret = ret.as_subclass(cls)
        if eps is not None:
            ret.eps = eps
    elif isinstance(ret, cls):
        if eps is not None:
            ret.eps = eps
    if isinstance(ret, (tuple, list)):
        # Also handles things like namedtuples
        ret = type(ret)(_convert(r, cls) for r in ret)
        #GEORGR: is this right?
        if eps is not None:
            for r in ret:
                if isinstance(r, QTensor):
                    r.eps = eps

    return ret

def qt_implements(torch_func : callable):
    def inner(f):
        QTensor.hookedMethods[torch_func.__name__] = f
        return f
    return inner


@qt_implements(torch.stack)
def qt_stack(tensors, dim=0, out=None):
    with torch._C.DisableTorchFunction():
        stacked = torch.stack(tensors, dim=dim, out=out)
    eps_out = None
    if all(isinstance(t, QTensor) and t.eps is not None for t in tensors):
        epses = [t.eps for t in tensors]
        eps_diffs = [(e1 - e2).abs().sum() for e1, e2 in zip(epses[:-1], epses[1:])]
        if not all(ed < 1e-8 for ed in eps_diffs):
            print("Warning: stacking QTensors  with different eps values!! Eps is discarded for resulting QTensor")
        else:
            eps_out = epses[0]
    stacked = _convert(stacked, QTensor, eps=eps_out)
    return stacked
