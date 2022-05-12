from .modulewisedescription import ModuleWiseDescription, ModuleWiseDescriptionSpecType
from .finder import ModuleWiseFinder
from .applier import ModuleWiseReplacer
from quantlib.editing.editing.editors import Rewriter
from quantlib.editing.graphs.fx import quantlib_symbolic_trace


class ModuleWiseConverter(Rewriter):

    def __init__(self, modulewisedescriptionspec: ModuleWiseDescriptionSpecType):

        self._modulewisedescription = ModuleWiseDescription(modulewisedescriptionspec)

        finder = ModuleWiseFinder(self.modulewisedescription)
        applier = ModuleWiseReplacer(self.modulewisedescription)

        super(ModuleWiseConverter, self).__init__('ModuleWiseConverter', quantlib_symbolic_trace, finder, applier)

    @property
    def modulewisedescription(self) -> ModuleWiseDescription:
        return self._modulewisedescription
