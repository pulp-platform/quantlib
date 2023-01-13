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

"""This module implements an abstraction to describe an ``nn.Module`` and
rich semantic checks on its composing sub-modules.

This abstraction, named ``NNModuleWithCheckers``, pairs:
  * an ``nn.Module``;
  * a dictionary mapping symbolic module names (i.e., strings) to collections
    of ``NNModuleChecker``s; an ``NNModuleChecker`` is a function taking in
    input an ``nn.Module`` object and returning a boolean expressing whether
    the object satisfies a given condition.
The interface that should be preserved by alternative ``NNModuleWithCheckers``
implementations is exposing these two (immutable) components as attributes:
  * ``module``;
  * ``name_to_checkers``.

Users of this Python module can describe ``NNModuleWithCheckers`` objects in
two ways.
  * An an ``NNModuleChecker``. The solver method is the identity function.
  * As an ``nn.Module``. The solver function will silently attach to each
    named sub-module a type checker, i.e., an ``NNModuleChecker`` verifying
    whether a given ``nn.Module`` is of the same type as the sub-module.
All these descriptions can be turned into ``NNModuleWithChecker``s by passing
them through the ``resolve_nnmodulewithcheckersspec`` canonicaliser.

"""

from enum import Enum
import torch.nn as nn
from typing import Tuple, Dict, Union, Callable, Type


# -- CANONICAL DATA STRUCTURE -- #

Checker = Callable[[nn.Module], bool]


CheckersMapType = Dict[str, Tuple[Checker, ...]]


class NNModuleWithCheckers(object):
    # TODO: make this a sub-class of `NamedTuple` with checks in the constructor

    def __init__(self,
                 module:           nn.Module,
                 name_to_checkers: Dict[str, Union[Checker, Tuple[Checker, ...]]]):

        # validate `module` argument
        if not isinstance(module, nn.Module):
            raise TypeError
        name_to_module = dict(module.named_modules())

        # validate `name_to_checkers` argument
        # keys (type)
        if not all(isinstance(name, str) for name in name_to_checkers.keys()):
            raise TypeError
        # keys (value)
        if not set(name_to_checkers.keys()).issubset(set(name_to_module.keys())):
            raise ValueError
        # values (type)
        if not all(callable(checkers) or (isinstance(checkers, tuple) and all(callable(c) for c in checkers)) for checkers in name_to_checkers.values()):
            raise TypeError
        # values (canonicalise)
        name_to_checkers = {name: checkers if (isinstance(checkers, tuple) and all(map(lambda c: callable(c), checkers))) else (checkers,) for name, checkers in name_to_checkers.items()}
        name_to_checkers = {name: (NNModuleWithCheckers._get_type_checker(type(pm)), *name_to_checkers.get(name, tuple())) for name, pm in name_to_module.items()}  # the first check should always be a type check
        # values (value - verify whether the target `nn.Module`s satisfy the conditions set by the checkers)
        for name, checkers in name_to_checkers.items():
            m = module.get_submodule(target=name)
            if not all(c(m) for c in checkers):
                raise ValueError

        # initialise components
        self._module: nn.Module = module
        self._name_to_checkers: CheckersMapType = name_to_checkers

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def name_to_checkers(self) -> CheckersMapType:
        return self._name_to_checkers

    @staticmethod
    def _get_type_checker(module_class: Type[nn.Module]):
        return lambda m: isinstance(m, module_class)


# -- CANONICALISATION FLOW -- #

NNModuleWithCheckersSpecType = Union[NNModuleWithCheckers, nn.Module]


def resolve_nnmodulewithcheckers_nnmodulewithcheckersspec(nnmodulewithcheckersspec: NNModuleWithCheckers) -> NNModuleWithCheckers:
    return nnmodulewithcheckersspec


def resolve_nnmodule_nnmodulewithcheckersspec(nnmodulewithcheckersspec: nn.Module) -> NNModuleWithCheckers:
    return NNModuleWithCheckers(module=nnmodulewithcheckersspec, name_to_checkers={})


ModuleWithCheckersSpecSolvers = Enum('ModuleWithCheckersSpec',
                                     [
                                         ('NNMODULEWITHCHECKERS', resolve_nnmodulewithcheckers_nnmodulewithcheckersspec),
                                         ('NN.MODULE',            resolve_nnmodule_nnmodulewithcheckersspec),
                                     ])


def resolve_nnmodulewithcheckersspec(nnmodulewithcheckersspec: NNModuleWithCheckersSpecType) -> NNModuleWithCheckers:

    # I apply a strategy pattern to retrieve the correct solver method based on the `ModuleWithCheckers` specification type
    modulewithcheckersspec_class = 'nn.Module'.upper() if isinstance(nnmodulewithcheckersspec, nn.Module) else nnmodulewithcheckersspec.__class__.__name__.upper()
    try:
        # solve the specification
        solver = getattr(ModuleWithCheckersSpecSolvers, modulewithcheckersspec_class)  # when the values of an enumerated are functions, I can not access them in dictionary-style: https://stackoverflow.com/a/50211710
        module_with_checkers = solver(nnmodulewithcheckersspec)
        return module_with_checkers
    except AttributeError:
        raise TypeError
