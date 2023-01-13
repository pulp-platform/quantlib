from .linearopintegeriser import *
from .requantiser import *
from .addrequantisationmerger import *

from quantlib.editing.editing.editors import ComposedEditor


class F2TIntegeriser(ComposedEditor):

    def __init__(self, B: int):
        super(F2TIntegeriser, self).__init__([
            LinearOpIntegeriser(),
            Requantiser(B),
        ])
