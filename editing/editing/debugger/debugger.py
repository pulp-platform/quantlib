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

import torch.fx as fx
import warnings
from typing import NamedTuple, List, Union, Optional, NewType

from ..editors.editors import Editor, Annotator, ApplicationPoint, Rewriter, BaseEditor
from quantlib.utils import quantlib_err_header, quantlib_wng_header


class AnnotationCommit(NamedTuple):
    """A structure to record a graph annotation.

    Annotating a graph is a non-destructive operation, since the topology of
    the target graph is not altered by the annotation. However, annotations
    modify the data structure holding the graph (e.g., an ``fx.GraphModule``),
    while we might want to consider the annotated and non-annotated graphs as
    distinct entities.
    """
    annotation: Annotator
    g:          fx.GraphModule


class RewritingCommit(NamedTuple):
    """A structure to record an atomic graph transformation.

    Starting from a given target graph, transforming it requires four steps:

    * select the graph rewriting rule;
    * find those sub-graphs of the target graph that match the rule's pattern,
      i.e., the rule's candidate application points;
    * amongst the candidate application points, choose one;
    * rewrite the chosen application point to get the modified graph.

    """
    rewriting:     Rewriter
    candidate_aps: List[ApplicationPoint]
    chosen_ap:     Optional[ApplicationPoint]
    g:             fx.GraphModule


# Both annotations and rewritings are required to transform a graph, so both
# operations concur to describe the processing flow.
Commit = NewType('Commit', Union[AnnotationCommit, RewritingCommit])


class History:
    """A structure to record the operations applied to a graph.

    Applying compositions of atomic transformations as a whole does not return
    handles to the intermediate graphs created during the process. To analyse
    such complex transforms, we need to record and access all the intermediate
    states. Note that it is important that these intermediate structures are
    all distinct from one another, i.e., that each transformation should not
    be applied in-place to the target graph, but to a copy structure.

    Example scenarios where this functionality is required are:

    * exploring experimental transformations; for instance, exploring the
      effect of applying a rewriting rule to different application points in
      different orders;
    * debugging long pipelines of transformations; for instance, when
      transforming computational graph, identifying those atomic transforms
      which introduce errors in the target graph;
    * analysing the effect of transformations on the functionality of the
      underlying computational graph; for instance, during fake-to-true
      conversions, we might want to analyse how the error propagates as
      fake-quantised ``QModule``s are mapped to true-quantised ones.

    This abstraction enables the analysis of complex, non-atomic graph
    transformations by recording sequences of atomic transformations applied
    to the target graph. Each atomic transformation is recorded as a
    ``Commit``, a tuple collecting pointers to the objects involved in the
    transformation.

    """

    def __init__(self):
        self._undo: List[Commit] = []
        self._redo: List[Commit] = []

    def __len__(self):
        """Count the atomic transformations required to reach the last state."""
        return len(self._undo)

    def __getitem__(self, i: int):
        """Make history subscriptable (i.e., accessible by ``[]`` notation)."""
        return self._undo[i]

    def push(self, c: Commit) -> None:
        self._undo.append(c)

    def _undo_one(self):
        try:
            c = self._undo.pop()
            self._redo.append(c)
        except IndexError:
            raise

    def undo(self, n: int = 1):
        for i in range(0, n):
            try:
                self._undo_one()
            except IndexError:
                raise IndexError(quantlib_err_header(obj_name=self.__class__.__name__) + f"tried to undo {n} Commits, but only had {i} in memory.")

    def _redo_one(self):
        try:
            c = self._redo.pop()
            self._undo.append(c)
        except IndexError:
            raise

    def redo(self, n: int = 1):
        for i in range(0, n):
            try:
                self._redo_one()
            except IndexError:
                raise IndexError(quantlib_err_header(obj_name=self.__class__.__name__) + f"tried to redo {n} Commits, but ony had {i} in memory.")


class Debugger:

    def __init__(self):
        self._editor:  Optional[BaseEditor] = None
        self._inplace: bool                 = True
        self._history: History              = History()

    def copybased(self) -> None:
        if self._inplace:
            self._inplace = False

    def inplace(self) -> None:
        if not self._inplace:
            self.flush_history()
            self._inplace = True

    def flush_history(self):
        """Release the current history to Python's garbage collector.

        We expose this function in ``Debugger``'s public interface since it
        might be necessary to manually flush the commit history (e.g., during
        a debugging sessions which is filling-up system memory).
        """
        self._history = History()

    def set_editor(self, editor: Editor) -> None:
        if not isinstance(editor, BaseEditor):
            raise TypeError(quantlib_err_header(obj_name=self.__class__.__name__) + f"can only set Annotators or Rewriters, but received {type(editor)}.")
        else:
            self._editor = editor

    @property
    def is_annotating(self) -> bool:
        return isinstance(self._editor, Annotator)

    def annotate(self, g: fx.GraphModule) -> fx.GraphModule:
        if self.is_annotating:
            g = self._editor.apply(g)
            return g
        else:
            warnings.warn(quantlib_wng_header(obj_name=self.__class__.__name__) + f"can not annotate when the current Editor is {self._editor} of type {type(self._editor)}.")

    @property
    def is_rewriting(self) -> bool:
        return isinstance(self._editor, Rewriter)

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        if self.is_rewriting:
            return self._editor.find(g)
        else:
            warnings.warn(quantlib_wng_header(obj_name=self.__class__.__name__) + f"can not find application points when the current Editor is {self._editor} of type {type(self._editor)}.")

    def rewrite(self, g: fx.GraphModule, aps: Union[None, ApplicationPoint, List[ApplicationPoint]] = None) -> fx.GraphModule:
        if self.is_rewriting:
            if self._inplace:
                pass
            else:
                # ERROR: I should have access to `aps` to update the pointed-to-`Node`s!
                # TODO: check that `ap` has been obtained using `self._edit`
                pass
            g = self._editor.apply(g, aps)
            return g
        else:
            warnings.warn(quantlib_wng_header(obj_name=self.__class__.__name__) + f"can not apply a Rewriting when Editor is {self._editor} of type {type(self._editor)}. Set a Rewriter first, recompute application points, finally apply the Rewriter.")
