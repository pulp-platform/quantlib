from .finder import EpsTunnelInserterFinder
from .applier import EpsTunnelInserterApplier
from quantlib.editing.editing.editors import Rewriter
from quantlib.editing.graphs.fx import quantlib_symbolic_trace


class EpsTunnelInserter(Rewriter):

    def __init__(self):
        super(EpsTunnelInserter, self).__init__('EpsTunnelInserter',
                                                quantlib_symbolic_trace,
                                                EpsTunnelInserterFinder(),
                                                EpsTunnelInserterApplier())
