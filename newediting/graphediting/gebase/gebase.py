from __future__ import annotations

import torch.fx as fx
from collections import OrderedDict
from typing import Union, List, Any, Callable

from quantlib.newutils import quantlib_err_header


class GraphEditor(object):

    def __init__(self):

        super(GraphEditor, self).__init__()

        self._parent_editor: Union[None, GraphEditor] = None
        self._children_editors: OrderedDict[str, GraphEditor] = OrderedDict()

    def _set_parent_editor(self, editor: GraphEditor) -> None:
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
        editor._set_parent_editor(self)

    def _apply(self, gm: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError

    def apply(self, gm: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError


class GraphAnnotator(GraphEditor):

    def __init__(self):
        super(GraphAnnotator, self).__init__()

    def register_child_editor(self, name: str, editor: GraphEditor) -> None:
        raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not register children GraphEditor objects.")

    def _apply(self, gm: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError

    def apply(self, gm: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError


class GraphRewriter(GraphEditor):

    def __init__(self, symbolic_trace_fun: Callable):
        super(GraphRewriter, self).__init__()
        self._symbolic_trace_fun = symbolic_trace_fun

    def _search_application_points(self, gm: fx.GraphModule) -> List[Any]:
        raise NotImplementedError

    def _retrace(self, gm: fx.GraphModule) -> fx.GraphModule:
        return self._symbolic_trace_fun(gm)

    def _polish(self, gm: fx.GraphModule) -> None:
        """Finalise the modifications applied by the ``GraphRewriter``."""
        gm.recompile()   # https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.recompile
        gm.graph.lint()  # https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint; this also enforces that the nodes appear in topological order

    def _apply(self, gm: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError

    def apply(self, gm: fx.GraphModule) -> fx.GraphModule:
        # gm = self._apply(gm)
        # self._polish(gm)
        # gm = self._retrace(gm)
        raise NotImplementedError
