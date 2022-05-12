from __future__ import annotations

import torch.fx as fx
from typing import NamedTuple, List, Union, Optional

from .applicationpoint import ApplicationPoint
from .finder import Finder
from .applier import Applier
from ..baseeditor import BaseEditor, SymbolicTraceFnType


# -- CONTEXTUALISED APPLICATION POINTS -- #

class Context(NamedTuple):

    rewriter: Rewriter
    graph:    fx.GraphModule

    def __eq__(self, other) -> bool:
        return isinstance(other, Context) and (self.rewriter is other.rewriter) and (self.graph is other.graph)


class ApplicationPointWithContext(NamedTuple):
    """Each application point must have been found by a specific ``Rewriter``
    on a specific ``fx.GraphModule``.
    """
    ap:      ApplicationPoint
    context: Context


# -- REWRITER -- #

class Rewriter(BaseEditor):
    """Base ``Editor`` representing a graph rewriting rule."""

    def __init__(self,
                 name:              str,
                 symbolic_trace_fn: SymbolicTraceFnType,
                 finder:            Finder,
                 applier:           Applier):

        super(Rewriter, self).__init__(name, symbolic_trace_fn)
        self._finder = finder
        self._applier = applier

    def find(self, g: fx.GraphModule) -> List[ApplicationPointWithContext]:
        aps = self._finder.find(g)  # find the application points
        apcontexts = list(map(lambda ap: ApplicationPointWithContext(ap=ap, context=Context(rewriter=self, graph=g)), aps))  # bind each application point to this `Rewriter` and the argument `fx.GraphModule`
        return apcontexts

    def apply(self, g: fx.GraphModule, apcontexts: Optional[Union[ApplicationPointWithContext, List[ApplicationPointWithContext]]] = None, *args, **kwargs) -> fx.GraphModule:

        # validate application point contexts argument
        # check type
        if not ((apcontexts is None) or isinstance(apcontexts, ApplicationPointWithContext) or (isinstance(apcontexts, list) and all(map(lambda apc: isinstance(apc, ApplicationPointWithContext), apcontexts)))):
            raise TypeError
        # canonicalise
        if apcontexts is None:
            apcontexts = self.find(g)
        elif isinstance(apcontexts, ApplicationPointWithContext):
            apcontexts = [apcontexts]
        # verify that the context is correct
        this_context = Context(rewriter=self, graph=g)
        if not all(map(lambda apc: apc.context == this_context, apcontexts)):
            raise ValueError
        # verify that the application points commute
        if not self._finder.check_aps_commutativity(list(map(lambda apc: apc.ap, apcontexts))):
            raise ValueError

        # rewrite all the application points
        for apc in apcontexts:
            g = self._applier.apply(g, apc.ap, self.id_)

        return g
