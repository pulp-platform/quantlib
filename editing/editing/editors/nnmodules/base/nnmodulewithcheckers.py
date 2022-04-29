"""This module implements an abstraction to describe an ``nn.Module`` and
descriptive checks on its composing sub-modules.

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
three ways:
  * An ``NNModuleChecker``.
  * As a list of ``NNModuleDescription``s (an ``NNModuleDescription`` is a
    tuple describing how to assemble a named ``nn.Module``). Since a list
    implicitly describes a chain relationship amongst its items, this
    description is syntactic sugar to create ``nn.Sequential`` objects. A
    single ``NNModuleDescription`` is also a valid description.
  * As an ``nn.Module``. The solver function will silently attach to each
    named sub-module a type checker, i.e., an ``NNModuleChecker`` verifying
    whether a given ``nn.Module`` is of the same type as the sub-module.
All these descriptions can be turned into ``NNModuleWithChecker``s by passing
them through the ``resolve_nnmodulewithcheckersspec`` canonicaliser.

"""

from collections import OrderedDict
from enum import Enum
import torch.nn as nn
from typing import NamedTuple, Tuple, List, Dict, Any, Union, Optional, Callable, Type


NNModuleChecker = Callable[[nn.Module], bool]


CheckersMapType = Dict[str, Union[NNModuleChecker, Tuple[NNModuleChecker, ...]]]


class NNModuleWithCheckers(object):

    def __init__(self,
                 module:           nn.Module,
                 name_to_checkers: CheckersMapType):

        # validate module
        if not isinstance(module, nn.Module):
            raise TypeError
        name_to_module = dict(module.named_children())

        # validate checker keys
        if not all(isinstance(name, str) for name in name_to_checkers.keys()):
            raise TypeError
        if not set(name_to_checkers.keys()).issubset(set(name_to_module.keys())):
            raise ValueError

        # (partially) validate checker tuples
        if not all(callable(checkers) or (isinstance(checkers, tuple) and all(callable(c) for c in checkers)) for checkers in name_to_checkers.values()):
            raise TypeError

        # canonicalise checker tuples
        name_to_checkers = {name: checkers if (isinstance(checkers, tuple) and all(map(lambda c: callable(c), checkers))) else (checkers,) for name, checkers in name_to_checkers.items()}
        name_to_checkers = {name: (NNModuleWithCheckers._get_type_checker(type(pm)), *name_to_checkers.get(name, tuple())) for name, pm in name_to_module.items()}

        # verify whether the argument module satisfies the checkers
        for name, checkers in name_to_checkers.items():
            m = module.get_submodule(target=name)
            if not all(c(m) for c in checkers):
                raise ValueError

        # initialise components
        self._module = module
        self._name_to_checkers = name_to_checkers

    @property
    def module(self) -> nn.Module:
        return self._module

    @property
    def name_to_checkers(self) -> CheckersMapType:
        return self._name_to_checkers

    @staticmethod
    def _get_type_checker(module_class: Type[nn.Module]):
        return lambda m: isinstance(m, module_class)


# -- SOLVER METHODS -- #

class NNModuleDescription(NamedTuple):
    name:     str
    class_:   Type[nn.Module]
    kwargs:   Dict[str, Any]
    checkers: Optional[Union[NNModuleChecker, Tuple[NNModuleChecker, ...]]] = None


NNSequentialDescription = Union[NNModuleDescription, List[NNModuleDescription]]


NNModuleWithCheckersSpecType = Union[NNModuleWithCheckers, NNSequentialDescription, nn.Module]


def resolve_nnmodulewithcheckers_nnmodulewithcheckersspec(nnmodulewithcheckersspec: NNModuleWithCheckers) -> NNModuleWithCheckers:
    return nnmodulewithcheckersspec


def resolve_nnsequentialdescription_nnmodulewithcheckersspec(nnmodulewithcheckersspec: NNSequentialDescription) -> NNModuleWithCheckers:

    # validate input type
    if not (isinstance(nnmodulewithcheckersspec, NNModuleDescription) or (isinstance(nnmodulewithcheckersspec, list) and all(isinstance(item_, NNModuleDescription) for item_ in nnmodulewithcheckersspec))):
        raise TypeError

    # canonicalise
    if isinstance(nnmodulewithcheckersspec, NNModuleDescription):
        nnmodulewithcheckersspec = [nnmodulewithcheckersspec]

    # create the object
    name_to_module = OrderedDict([(desc.name, desc.class_(**desc.kwargs)) for desc in nnmodulewithcheckersspec])
    module = nn.Sequential(name_to_module)
    name_to_checkers = {desc.name: desc.checkers for desc in nnmodulewithcheckersspec if desc.checkers is not None}
    module_with_checkers = NNModuleWithCheckers(module=module, name_to_checkers=name_to_checkers)

    return module_with_checkers


def resolve_nnmodule_nnmodulewithcheckersspec(nnmodulewithcheckersspec: nn.Module) -> NNModuleWithCheckers:
    return NNModuleWithCheckers(module=nnmodulewithcheckersspec, name_to_checkers={})


ModuleWithCheckersSpecSolvers = Enum('ModuleWithCheckersSpec',
                                     [
                                         ('NNMODULEWITHCHECKERS', resolve_nnmodulewithcheckers_nnmodulewithcheckersspec),
                                         ('NNMODULEDESCRIPTION',  resolve_nnsequentialdescription_nnmodulewithcheckersspec),
                                         ('LIST',                 resolve_nnsequentialdescription_nnmodulewithcheckersspec),
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
