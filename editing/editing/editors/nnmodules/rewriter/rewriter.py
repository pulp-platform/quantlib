from ..pattern.base import NNModulePattern
from ..finder.base import NNModuleMatcher
from ..applier import NNModuleApplier
from quantlib.editing.editing.editors.base import Rewriter


class NNModuleRewriter(Rewriter):

    def __init__(self,
                 name:    str,
                 pattern: NNModulePattern,
                 finder:  NNModuleMatcher,
                 applier: NNModuleApplier):

        if not ((pattern is finder.pattern) and (pattern is applier.pattern)):
            raise ValueError

        super(NNModuleRewriter, self).__init__(name, pattern.symbolic_trace_fn, finder, applier)
        self._pattern = pattern

    @property
    def pattern(self) -> NNModulePattern:
        return self._pattern
