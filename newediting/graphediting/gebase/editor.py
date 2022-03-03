import torch.fx as fx
from typing import Optional, Any, NamedTuple, List, Union, NewType

from .annotation import Annotation
from .rewriting import Rewriting
from quantlib.newutils import quantlib_err_header


class AnnotationCommit(NamedTuple):
    """A structure to record a graph annotation.

    Annotating a graph is a non-destructive operation, since the topology of
    the target graph is not altered by the annotation. However, annotations
    modify the data structure holding the graph (e.g., an ``fx.GraphModule``),
    while we might want to consider the annotated and non-annotated graphs as
    distinct entities.
    """
    annotation: Annotation
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
    rewriting:     Rewriting
    candidate_aps: List[Any]
    chosen_ap:     Any
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


class Editor:

    def __init__(self):
        self._inplace:    bool = True
        self._history:    History = History()
        self._rewriting:  Optional[Any] = None
        self._annotation: Optional[Any] = None

    def copybased(self) -> None:

        if self._inplace:
            self._inplace = False

        else:
            warnings.warn()

    def inplace(self) -> None:

        if self._inplace:
            warnings.warn()

        else:
            self.flush_history()
            self._inplace = True

    def flush_history(self):
        """Release the current history to Python's garbage collector.

        We expose this function in ``Editor``'s public interface since it is
        convenient to manually flush the commit history.
        """
        self._history = History()

    def set_rewriting(self, rho: Rewriting) -> None:
        raise NotImplementedError

    def find(self, g: fx.GraphModule) -> List[Any]:
        raise NotImplementedError

    def rewrite(self, g: fx.GraphModule, ap: Any) -> fx.GraphModule:
        raise NotImplementedError

    def rewrite_all(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError

    def set_annotation(self, alpha: Annotation) -> None:
        raise NotImplementedError

    def annotate(self, g: fx.GraphModule) -> fx.GraphModule:
        raise NotImplementedError
