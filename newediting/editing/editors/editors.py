from __future__ import annotations

import torch.fx as fx
from typing import NamedTuple, List, Dict, Union, Optional

from quantlib.newediting.graphs import Node as LWNode
from quantlib.newutils import quantlib_err_header


class Editor(object):

    def __init__(self):
        self._parent_editor: Optional[Editor] = None

    def set_parent_editor(self, editor: Editor) -> None:
        self._parent_editor = editor

    @property
    def parent_editor(self) -> Editor:
        return self._parent_editor

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError

    def __call__(self, g: fx.GraphModule) -> fx.GraphModule:
        return self.apply(g)


_QL_EDITOR_PREFIX = '_QL'


class Annotator(Editor):
    """Base ``Editor`` representing an annotation.

    Its application does not change the topology of the graph, but only its
    attributes (i.e., it modifies or enriches the semantics of the graph).

    """

    def __init__(self, name: str):
        super(Annotator, self).__init__()
        # we use the following attribute to identify the annotations made by the `Annotator`
        self._name: str = '_'.join([_QL_EDITOR_PREFIX, name, str(id(self))])

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError


ApplicationPointCore = Union[LWNode, fx.Node, Dict[fx.Node, fx.Node]]


class ApplicationPoint(NamedTuple):
    rewriter: Rewriter
    graph:    fx.GraphModule
    apcore:   ApplicationPointCore


class Rewriter(Editor):
    """Base ``Editor`` representing a graph rewriting rule.

    Each application is an atomic topological edit of the target graph.

    """

    def __init__(self, name: str):

        super(Rewriter, self).__init__()

        # we use the following two attributes to annotate all the transformations made by the `Rewriter`
        self._name:    str = '_'.join([_QL_EDITOR_PREFIX, name, str(id(self))])
        self._counter: int = 0

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        # Perform the following:
        #   - find the `ApplicationPointCore`s;
        #   - bind the cores to the `Rewriter` instance that found them, and to the `fx.GraphModule` used as a data graph.
        raise NotImplementedError

    def _check_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:
        # Check that:
        #   - the application points were computed by this rule on the target graph;
        #   - the application points are independent of each other, so that the associated rewritings commute.
        raise NotImplementedError

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:
        """Apply the rewriting to the target ``fx.GraphModule`` at the
        argument ``ApplicationPoint``.

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

        # canonicalise application points argument
        if aps is None:
            aps = self.find(g)
        elif isinstance(aps, ApplicationPoint):
            aps = [aps]
        elif isinstance(aps, list) and all(map(lambda ap: isinstance(ap, ApplicationPoint), aps)):
            pass
        else:
            raise TypeError

        # ensure that application all the application points can be rewritten independently of each other, so that the associated rewritings commute (e.g., their supports do not intersect)
        self._check_aps(g, aps)

        # rewrite all the application points
        for ap in aps:
            g = self._apply(g, ap)
            Rewriter._polish_fxgraphmodule(g)

        return g


BaseEditor = Union[Annotator, Rewriter]


class ComposedEditor(Editor):
    """``Editor`` applying a sequence of editing steps to the target graph."""

    def __init__(self,
                 editors: List[Editor]):

        super(ComposedEditor, self).__init__()

        self._children_editors: List[Editor] = []
        for editor in editors:
            self._register_child_editor(editor)

    def _register_child_editor(self, editor: Editor) -> None:
        """Register a child ``Editor`` with the current ``Editor``.

        This function has two side-effects:

        * the argument ``Editor`` will be added to the collection of children
          of the current ``Editor``;
        * the argument ``Editor`` will have its parent editor set to the
          current ``Editor``.

        The overall effect is therefore setting two-way references between an
        ``Editor`` and its children.

        The intended purpose of the function is two-fold: guaranteeing that
        each ``Editor`` has at most one parent, and guaranteeing that the
        order of the children is deterministic. These two properties ensure
        that:

        * each ``Editor`` is called in the scope of at most one parent
          ``Editor``, and therefore analysing the call stack of a single
          ``Editor`` returns complete information about the evolution of the
          state of its children ``Editor``s;
        * that the call stack of a single ``Editor`` should be consistent
          across different executions (in the hypothesis of processing the
          same ``fx.GraphModule``).

        """

        if editor.parent_editor is not None:
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not register a children Editor when it already has a parent Editor. Create and register a new instance of the same Editor.")

        else:
            self._children_editors.append(editor)
            editor.set_parent_editor(self)

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:

        for editor in self._children_editors:
            g = editor(g)

        return g
