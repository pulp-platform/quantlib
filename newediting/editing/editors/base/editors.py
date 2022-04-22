"""
This module implements the basic class hierarchy of QuantLib's graph editing
machinery. Edits are partitioned into basic edits and composed edits. Basic
edits are further partitioned into annotations (which add attributes and other
semantic information but do not modify existing graph information) and
rewritings (which can modify existing graph information, and even change the
topology of the graph).
"""

from __future__ import annotations

import torch.fx as fx
from typing import NamedTuple, List, Dict, Union

from quantlib.newediting.graphs import Node as LWNode
from quantlib.newutils import quantlib_err_header


class Editor(object):

    def __init__(self):
        super(Editor, self).__init__()

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError

    def __call__(self, g: fx.GraphModule) -> fx.GraphModule:
        return self.apply(g)


# -- BASE EDITORS -- #

class BaseEditor(Editor):

    def __init__(self, name: str):
        super(BaseEditor, self).__init__()
        self._editor_id: str = '_'.join(['_QL', name, str(id(self))])  # this attribute allows to uniquely identify the edits due to this `Editor`

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError


class Annotator(BaseEditor):
    """Base ``Editor`` representing an annotation.

    Its application does not change the topology of the graph, but only its
    attributes (i.e., it modifies or enriches the semantics of the graph).

    """

    def __init__(self, name: str):
        super(Annotator, self).__init__(name)

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError


ApplicationPointCore = Union[LWNode, fx.Node, Dict[fx.Node, fx.Node]]


class ApplicationPoint(NamedTuple):
    rewriter: Rewriter
    graph:    fx.GraphModule
    core:     ApplicationPointCore


class Rewriter(BaseEditor):
    """Base ``Editor`` representing a graph rewriting rule.

    Each application is an atomic topological edit of the target graph.

    """

    def __init__(self, name: str):

        super(Rewriter, self).__init__(name)
        self._counter: int = 0  # we use this attribute to annotate all the transformations made by the `Rewriter`

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        # Perform the following:
        #   - find the `ApplicationPointCore`s;
        #   - bind the cores to the `Rewriter` instance that found them, and to the `fx.GraphModule` used as a data graph.
        raise NotImplementedError

    def _canonicalise_aps(self, g: fx.GraphModule, aps: Union[None, ApplicationPoint, List[ApplicationPoint]] = None) -> List[ApplicationPoint]:

        if aps is None:
            aps = self.find(g)
        elif isinstance(aps, ApplicationPoint):
            aps = [aps]
        elif isinstance(aps, list) and all(map(lambda ap: isinstance(ap, ApplicationPoint), aps)):
            pass
        else:
            raise TypeError

        return aps

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        """Verify whether the order of application of the rewritings matters.

        Rule programmers can implement here their decisions on how to handle
        non-commutative rewriting rules. Usual choices might be to reject
        non-singleton lists of application points, or warning the user that
        an arbitrary application order will be used.
        """
        raise NotImplementedError

    def _validate_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:

        # check that the application points were computed by this rule on the target graph
        if not all(map(lambda ap: (ap.rewriter is self) and (ap.graph is g), aps)):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not be applied to application points computed by other `Rewriter`s.")

        # check that the application points are independent of each other, so that the associated rewritings can commute
        self._check_aps_independence(aps)

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:
        """Apply the rewriting to the target ``fx.GraphModule`` at the given
        ``ApplicationPoint``.

        Note that this function operates by side-effect, i.e., by modifying
        the argument ``fx.GraphModule``.
        """
        raise NotImplementedError

    @staticmethod
    def _polish_fxgraphmodule(g: fx.GraphModule) -> None:
        """Finalise the modifications applied by the ``Rewriter``."""
        g.graph.lint()  # https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint; this also ensures that `fx.Node`s appear in topological order
        g.recompile()   # https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.recompile

    def apply(self, g: fx.GraphModule, aps: Union[None, ApplicationPoint, List[ApplicationPoint]] = None) -> fx.GraphModule:

        # canonicalise and validate the application points argument
        aps = self._canonicalise_aps(g, aps)
        self._validate_aps(g, aps)

        # rewrite all the application points
        for ap in aps:
            g = self._apply(g, ap)
            Rewriter._polish_fxgraphmodule(g)

        return g


# -- COMPOSED EDITOR -- #

class ComposedEditor(Editor):
    """``Editor`` applying a sequence of editing steps to the target graph."""

    def __init__(self, children_editors: List[Editor]):
        super(ComposedEditor, self).__init__()
        self._children_editors = children_editors

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:

        for editor in self._children_editors:
            g = editor(g)

        return g
