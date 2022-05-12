from typing import Union, Type

from ..pattern import NNSequentialPattern, GenericNNModulePattern
from ..finder import PathGraphMatcher, GenericGraphMatcher
from ..applier import NNModuleApplier
from .rewriter import NNModuleRewriter


def get_rewriter_class(name:          str,
                       pattern:       Union[NNSequentialPattern, GenericNNModulePattern],
                       finder_class:  Type[Union[PathGraphMatcher, GenericGraphMatcher]],
                       applier_class: Type[NNModuleApplier]) -> Type[NNModuleRewriter]:
    """A utility to generate ``NNModuleRewriter`` classes programmatically."""

    # validate input types
    if not isinstance(name, str):
        raise TypeError
    if not isinstance(pattern, (NNSequentialPattern, GenericNNModulePattern,)):
        raise TypeError
    if not isinstance(finder_class, (type(PathGraphMatcher), type(GenericGraphMatcher),)):
        raise TypeError
    if not isinstance(applier_class, type(NNModuleApplier)):
        raise TypeError

    # create the `Finder`
    finder = finder_class(pattern)

    # create the `Applier`
    applier = applier_class(pattern)

    # sub-class `Rewriter` around these specific `Finder` and `Applier`
    def __init__(self_):
        NNModuleRewriter.__init__(self_, name, pattern, finder, applier)

    rewriter_class = type(name, (NNModuleRewriter,), {'__init__': __init__})

    return rewriter_class
