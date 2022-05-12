from .tracing import quantlib_harmonisedadd_symbolic_trace
from .finder import AddTreeFinder
from .applier import AddTreeApplier
from ..qdescription import QDescriptionSpecType
from quantlib.editing.editing.editors import Rewriter


class AddTreeHarmoniser(Rewriter):

    def __init__(self,
                 qdescriptionspec: QDescriptionSpecType,
                 use_output_eps:   bool):

        finder = AddTreeFinder()
        applier = AddTreeApplier(qdescriptionspec, use_output_eps)

        super(AddTreeHarmoniser, self).__init__('AddTreeHarmoniser', quantlib_harmonisedadd_symbolic_trace, finder, applier)
