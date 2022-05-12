from .finder import EpsTunnelConstructFinder
from .applier import EpsTunnelConstructApplier
from quantlib.editing.editing.editors import Rewriter
from quantlib.editing.graphs.fx import quantlib_symbolic_trace


class EpsTunnelConstructSimplifier(Rewriter):

    def __init__(self):
        super(EpsTunnelConstructSimplifier, self).__init__('EpsTunnelConstructSimplifier',
                                                           quantlib_symbolic_trace,
                                                           EpsTunnelConstructFinder(),
                                                           EpsTunnelConstructApplier())
