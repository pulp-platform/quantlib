"""
This module implements the basic class hierarchy of QuantLib's graph editing
machinery. Edits are partitioned into basic edits and composed edits. Basic
edits are further partitioned into annotations (which add attributes and other
semantic information but do not modify functional graph information) and
rewritings (which can modify functional graph information, and even change the
topology of the graph).
"""

from __future__ import annotations

from abc import ABC
import torch.fx as fx
from typing import NamedTuple, List, Dict, Union, Optional

from ....graphs.fx import SymbolicTraceFnType


#                      Editor                                                    #
#                        /\                                                      #
#                       /  \                                                     #
#          BaseEditor _/    \_ ComposedEditor                                    #
#              /\                                                                #
#             /  \                                                               #
#  Annotator_/    \_ Rewriter                                                    #
#                    |_ ApplicationPoint + Context = ApplicationPointWithContext #
#                    |_ Finder                                                   #
#                    |_ Applier                                                  #


# -- 0. BASE CLASS -- #

class Editor(object):

    def __init__(self):
        super(Editor, self).__init__()

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError

    def __call__(self, g: fx.GraphModule) -> fx.GraphModule:
        return self.apply(g)


# -- 1. BASE EDITORS -- #

class BaseEditor(Editor):

    def __init__(self,
                 name:              str,
                 symbolic_trace_fn: SymbolicTraceFnType):

        super(BaseEditor, self).__init__()

        self._id: str = '_'.join(['QL', name, str(id(self))])  # we use this attribute to uniquely identify the edits made using this `Editor`
        self._symbolic_trace_fn = symbolic_trace_fn            # we assume that the `fx.GraphModule`s processed by this `Editor` have been obtained using this tracing function

    @property
    def id(self) -> str:
        return self._id

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError


# -- 1a. Annotator -- #

class Annotator(BaseEditor):
    """Base ``Editor`` representing an annotation.

    Its application does not change the topology of the graph, but only its
    attributes (i.e., it modifies or enriches the semantics of the graph).

    """

    def __init__(self,
                 name:              str,
                 symbolic_trace_fn: SymbolicTraceFnType):
        super(Annotator, self).__init__(name, symbolic_trace_fn)

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError


# -- 1b. Rewriter -- #

ApplicationPoint = type('ApplicationPoint', (ABC,), {})  # in this way, we let each `Rewriter` define what an `ApplicationPoint` is for it


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


class Finder(object):

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        raise NotImplementedError

    def check_aps_commutativity(self, aps: List[ApplicationPoint]) -> bool:
        """Verify that the application points do not overlap.

        Passing this test ensures that the rewritings of the different
        application points can commute. Therefore, this avoids the need of
        recomputing the application points in-between applications of the same
        ``Rewriter``.

        """
        raise NotImplementedError


class Applier(object):

    def __init__(self):
        self._context_counters: Dict[Context, int] = {}

    def application_id(self, apc: ApplicationPointWithContext) -> str:
        return '_'.join([apc.context.rewriter.id, str(self._context_counters[apc.context])])

    def _apply(self, g: fx.GraphModule, apc: ApplicationPointWithContext) -> fx.GraphModule:
        # use `self.application_id(apc)` to mark the rewritten graph components
        raise NotImplementedError

    @staticmethod
    def _polish_fxgraphmodule(g: fx.GraphModule) -> None:
        """Finalise the modifications made in ``_apply``."""
        g.graph.lint()  # https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint; this also ensures that `fx.Node`s appear in topological order
        g.recompile()   # https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.recompile

    def apply(self, g: fx.GraphModule, apc: ApplicationPointWithContext) -> fx.GraphModule:

        # update the context counters
        try:
            self._context_counters[apc.context] += 1
        except KeyError:
            self._context_counters[apc.context] = 0

        # modify the graph
        g = self._apply(g, apc)
        Applier._polish_fxgraphmodule(g)

        return g


class Rewriter(BaseEditor):
    """Base ``Editor`` representing a graph rewriting rule.

    Each application is an atomic topological edit of the target graph.

    """

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

    def apply(self, g: fx.GraphModule, apcontexts: Optional[Union[ApplicationPointWithContext, List[ApplicationPointWithContext]]] = None) -> fx.GraphModule:

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
        if not all(map(lambda apc: apc.context is this_context, apcontexts)):
            raise ValueError
        # verify that the application points commute
        if not self._finder.check_aps_commutativity(list(map(lambda apc: apc.ap, apcontexts))):
            raise ValueError

        # rewrite all the application points
        for apc in apcontexts:
            g = self._applier.apply(g, apc)

        return g


# -- 2. COMPOSED EDITOR -- #

class ComposedEditor(Editor):
    """``Editor`` applying a sequence of editing steps to the target graph."""

    def __init__(self, children_editors: List[Editor]):

        # validate input
        if not (isinstance(children_editors, list) and all(map(lambda editor: isinstance(editor, Editor), children_editors))):
            raise TypeError

        super(ComposedEditor, self).__init__()
        self._children_editors = children_editors

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:

        for editor in self._children_editors:
            g = editor(g)

        return g
