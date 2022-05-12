import torch.fx as fx
from typing import List

from .editor import Editor


class ComposedEditor(Editor):
    """``Editor`` applying a sequence of editing steps to the target graph."""

    def __init__(self, children_editors: List[Editor]):

        # validate input
        if not (isinstance(children_editors, list) and all(map(lambda editor: isinstance(editor, Editor), children_editors))):
            raise TypeError

        super(ComposedEditor, self).__init__()
        self._children_editors = children_editors

    def apply(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:

        for editor in self._children_editors:
            g = editor(g, *args, **kwargs)

        return g
