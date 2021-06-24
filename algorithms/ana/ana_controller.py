# 
# ana_controller.py
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

from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn

from .ana_ops import ANAModule
from ..controller import Controller

import typing
from typing import Union, List


def lws(t: int, tstart: int, tend: int, alpha: int) -> float:
    """Limited-window scheduling function."""

    # assert 0 <= tstart < tend
    # assert 0 <= alpha

    t = min(max(t, tstart), tend)
    return (float(tend - t) / float(tend - tstart)) ** alpha


def uws(t: int, tstart: int, eps: float, alpha: int) -> float:
    """Unlimited-window scheduling function."""

    # assert 0 <= tstart
    # assert 0.0 <= eps
    # assert 0 <= alpha

    t          = max(t, tstart)
    multiplier = (1 / float(t - tstart + 1)) ** alpha
    multiplier = 0.0 if multiplier < eps else multiplier

    return multiplier


_AC_MAPPER = {'lws': lws, 'uws': uws}


class ANATimer(object):

    def __init__(self, fmi_spec: dict, fsigma_spec: dict, bmi_spec: dict, bsigma_spec: dict):

        self._fmi = None
        assert ANATimer.check_spec(fmi_spec, is_sigma=False)
        self._fmi_base = fmi_spec['base']
        self._fmi_fun  = partial(_AC_MAPPER[fmi_spec['fun']], **fmi_spec['kwargs'])

        self._fsigma = None
        assert ANATimer.check_spec(fsigma_spec, is_sigma=True)
        self._fsigma_base = fsigma_spec['base']
        self._fsigma_fun  = partial(_AC_MAPPER[fsigma_spec['fun']], **fsigma_spec['kwargs'])

        self._bmi = None
        assert ANATimer.check_spec(bmi_spec, is_sigma=False)
        self._bmi_base = bmi_spec['base']
        self._bmi_fun  = partial(_AC_MAPPER[bmi_spec['fun']], **bmi_spec['kwargs'])

        self._bsigma = None
        assert ANATimer.check_spec(bsigma_spec, is_sigma=True)
        self._bsigma_base = bsigma_spec['base']
        self._bsigma_fun  = partial(_AC_MAPPER[bsigma_spec['fun']], **bsigma_spec['kwargs'])

    @property
    def fmi(self) -> Union[None, float]:
        return self._fmi

    @property
    def fsigma(self) -> Union[None, float]:
        return self._fsigma

    @property
    def bmi(self) -> Union[None, float]:
        return self._bmi

    @property
    def bsigma(self) -> Union[None, float]:
        return self._bsigma

    @staticmethod
    def check_spec(spec: dict, is_sigma: bool) -> bool:

        check_base = isinstance(spec['base'], float) and (True if not is_sigma else (0.0 <= spec['base']))
        spec_type  = spec['fun']
        check_fun  = spec_type in set(_AC_MAPPER.keys())

        if check_base and check_fun:
            spec_kwargs = spec['kwargs']
            if spec_type == 'lws':
                check_tstart = isinstance(spec_kwargs['tstart'], int) and (0 <= spec_kwargs['tstart'])
                check_tend   = isinstance(spec_kwargs['tend'], int)   and (spec_kwargs['tstart'] < spec_kwargs['tend'])
                check_alpha  = isinstance(spec_kwargs['alpha'], int)  and (0 <= spec_kwargs['alpha'])
                is_correct = check_tstart and check_tend and check_alpha
            elif spec_type == 'uws':
                check_tstart = isinstance(spec_kwargs['tstart'], int) and (0 <= spec_kwargs['tstart'])
                check_eps    = isinstance(spec_kwargs['eps'], float)  and (0.0 <= spec_kwargs['eps'])
                check_alpha  = isinstance(spec_kwargs['alpha'], int)  and (0 <= spec_kwargs['alpha'])
                is_correct   = check_tstart and check_eps and check_alpha
            else:
                raise NotImplemented
        else:
            is_correct = False

        return is_correct

    def step(self, t: int) -> None:

        self._fmi    = self._fmi_base    * self._fmi_fun(t)
        self._fsigma = self._fsigma_base * self._fsigma_fun(t)

        self._bmi    = self._bmi_base    * self._bmi_fun(t)
        self._bsigma = self._bsigma_base * self._bsigma_fun(t)


Timer2Modules = typing.NamedTuple('Timer2Modules', [('timer', ANATimer), ('modules', List[torch.nn.Module])])


class ANAController(Controller):
    """The training controller for the *additive noise annealing* algorithm.

    This object sets and updates the hyper-parameters of the additive noise
    annealing (ANA) algorithm. These hyper-parameters are the means and the
    standard deviations of uni-variate probability distributions describing
    random variables that are added to the arguments of the quantizers. The
    ANA algorithm achieves quantization by gradually *annealing* these
    distributions to Dirac deltas centered at zero; the annealing has to be
    intended as convergence in distribution.

    For simplicity, ANA ``torch.nn.Module``s add the same noise to all the
    components of their input ``torch.Tensor``s. This reduces the number of
    hyper-parameters required by each ANA module to two (the mean and the
    standard deviation of the distribution). However, the ANA algorithm can
    be interpreted as a generalisation of the commonly used *straight-through
    estimator* (STE) algorithm and of other algorithms if we allow it to use
    different noise distributions during the forward and backward passes of
    the training algorithm. Therefore, each ANA module is attached four
    hyper-parameters: two for the forward pass, and two for the backward pass.
    """

    def __init__(self, module: torch.nn.Module, ctrl_spec: List) -> None:

        is_nndataparallel_module = isinstance(module, nn.DataParallel)

        self._global_step   = -1
        self._timer2modules = []

        self._n2m = self.get_modules(module)
        self._n2m = {('module.' if is_nndataparallel_module else '') + k: v for k, v in self._n2m.items()}

        # verify that each ANA module is linked to exactly one timer (i.e., check that ANA targets form a partition of the collection of ANA modules)
        all_ana_module_names = set(self._n2m.keys())
        ana_timer_targets    = []
        for timer_spec in ctrl_spec:
            ana_timer_target = set([('module.' if is_nndataparallel_module else '') + m for m in timer_spec['modules']])
            assert ana_timer_target.issubset(all_ana_module_names)  # non-ANA module specified as ANA target
            ana_timer_targets.append(ana_timer_target)
        assert len(all_ana_module_names) == sum([len(ana_timer_target) for ana_timer_target in ana_timer_targets])
        assert len(all_ana_module_names) == len(set.union(*ana_timer_targets))

        # build timers and link them to the specified ANA modules
        for timer_spec in ctrl_spec:
            timer        = ANATimer(timer_spec['fnoise']['mi'], timer_spec['fnoise']['sigma'], timer_spec['bnoise']['mi'], timer_spec['bnoise']['sigma'])
            modules      = list(map(lambda n: self._n2m[n], set([('module.' if is_nndataparallel_module else '') + m for m in timer_spec['modules']])))
            self._timer2modules.append(Timer2Modules(timer=timer, modules=modules))

    @staticmethod
    def get_modules(module: torch.nn.Module, parent_name: str = '', n2m: OrderedDict = OrderedDict()):

        for n, m in module.named_children():
            if len(list(m.children())) == 0:  # ``Module`` is not ``nn.Sequential`` or other container type; i.e., this is a "leaf" ``Module``
                if isinstance(m, ANAModule):
                    n2m.update({parent_name + n: m})
            else:
                ANAController.get_modules(m, parent_name=''.join([parent_name, n, '.']), n2m=n2m)

        return n2m

    @property
    def n2m(self) -> dict:
        return self._n2m

    def state_dict(self) -> dict:
        return {'_global_step': self._global_step}

    def step_pre_training_epoch(self, *args, **kwargs) -> None:

        self._global_step += 1

        for timer, modules in self._timer2modules:

            timer.step(self._global_step)

            for m in modules:
                # update forward noise
                m.set_fnoise(timer.fmi, timer.fsigma)
                # update backward noise
                m.set_bnoise(timer.bmi, timer.bsigma)

    def step_pre_validation(self, *args, **kwargs) -> None:

        for timer, modules in self._timer2modules:

            for m in modules:
                # remove forward noise
                m.set_fnoise(0.0, 0.0)
                # # remove backward noise
                # m.set_bnoise(0.0, 0.0)

