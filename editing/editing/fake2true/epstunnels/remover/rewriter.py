from .finder import EpsTunnelRemoverFinder
from .applier import EpsTunnelRemoverApplier
from quantlib.editing.editing.editors import Rewriter
from quantlib.editing.graphs.fx import quantlib_symbolic_trace


class EpsTunnelRemover(Rewriter):

    def __init__(self):
        super(EpsTunnelRemover, self).__init__('EpsTunnelRemover',
                                               quantlib_symbolic_trace,
                                               EpsTunnelRemoverFinder(),
                                               EpsTunnelRemoverApplier())
