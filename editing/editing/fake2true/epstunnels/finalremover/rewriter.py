from .finder import FinalEpsTunnelRemoverFinder
from .applier import FinalEpsTunnelRemoverApplier
from quantlib.editing.editing.editors import Rewriter
from quantlib.editing.graphs.fx import quantlib_symbolic_trace

# removes the output node eps-tunnel
class FinalEpsTunnelRemover(Rewriter):

    def __init__(self):
        super(FinalEpsTunnelRemover, self).__init__('FinalEpsTunnelRemover',
                                               quantlib_symbolic_trace,
                                               FinalEpsTunnelRemoverFinder(),
                                               FinalEpsTunnelRemoverApplier())
