from __future__ import annotations

import torch.fx as fx
from collections import OrderedDict
from typing import Union, Optional, List, Any, NewType

from quantlib.newutils import quantlib_err_header


class GraphEditor(object):

    def __init__(self):

        super(GraphEditor, self).__init__()

        self._parent_editor: Union[None, GraphEditor] = None
        self._children_editors: OrderedDict[str, GraphEditor] = OrderedDict()

    def set_parent_editor(self, editor: GraphEditor) -> None:
        self._parent_editor = editor

    @property
    def parent_editor(self) -> GraphEditor:
        return self._parent_editor

    def register_child_editor(self, name: str, editor: GraphEditor) -> None:
        """Register a child ``GraphEditor`` with the current ``GraphEditor``.

        This function has two side-effects:

        * the provided ``GraphEditor`` will be added to the collection of
          children of the current ``GraphEditor``;
        * the new child ``GraphEditor`` will have its parent editor set to the
          current ``GraphEditor``.

        The overall effect is therefore setting two-way references between an
        editor and its children.

        The intended purpose of the function is two-fold: guaranteeing that
        each editor has at most one parent, and guaranteeing that the order of
        the children is deterministic. These two properties ensure that:

        * each editor is called in the scope of at most one parent editor, and
          therefore analysing the call stack of a single ``GraphEditor``
          returns complete information about the evolution of the state of its
          children editors;
        * that the call stack of a single ``GraphEditor`` should be consistent
          across different executions (in the hypothesis of processing the
          same ``fx.GraphModule``).

        """

        if editor.parent_editor is not None:
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not register a candidate children GraphEditor when the candidate already has a parent GraphEditor.")

        self._children_editors[name] = editor
        editor.set_parent_editor(self)

    def apply(self, data_gm: fx.GraphModule):
        raise NotImplementedError


class GraphAnnotator(GraphEditor):

    def __init__(self):
        super(GraphAnnotator, self).__init__()

    def register_child_editor(self, name: str, editor: GraphEditor) -> None:
        raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not register children GraphEditor objects.")

    def apply(self, gm: fx.GraphModule):
        raise NotImplementedError


ApplicationPoint = NewType('ApplicationPoint', Any)


class GraphRewriter(GraphEditor):

    def __init__(self, name: str):

        super(GraphRewriter, self).__init__()

        # we use the following two parameters to annotate all the transformations made by the `GraphRewriter`
        self._name: str = '_QL_' + name
        self._counter: int = 0

    def register_child_editor(self, name: str, editor: GraphEditor) -> None:
        raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not register children GraphEditor objects.")

    def _polish_graphmodule(self, data_gm: fx.GraphModule) -> None:
        """Finalise the modifications applied by the ``GraphRewriter``."""
        data_gm.recompile()   # https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.recompile
        data_gm.graph.lint()  # https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint; this also enforces that the nodes appear in topological order

    def find_application_points(self, data_gm: fx.GraphModule) -> List[Any]:
        raise NotImplementedError

    def _select_ap(self, data_gm: fx.GraphModule) -> ApplicationPoint:

        all_application_points = self.find_application_points(data_gm)
        try:
            ap = next(iter(all_application_points))
        except StopIteration:
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "the GraphRewriter could not detect any application point.")

        return ap

    def _apply(self, data_gm: fx.GraphModule, ap: ApplicationPoint):
        """Apply the rewriting to the target ``fx.GraphModule`` on the
        provided application point.

        Note that this function operates by side-effect, i.e., by modifying
        the provided ``fx.GraphModule``.
        """
        # [...]
        # self._polish_graphmodule(data_gm)
        raise NotImplementedError

    def apply(self, data_gm: fx.GraphModule, ap: Optional[ApplicationPoint] = None):
        # # if no specific application point is provided, select the first application point found by the `GraphWriter`'s automatic procedure
        # if ap is None:
        #     ap = self._select_ap(data_gm)
        # self._apply(data_gm)
        raise NotImplementedError
