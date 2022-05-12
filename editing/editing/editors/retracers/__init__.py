from .retracer import Retracer
from quantlib.editing.graphs.fx import quantlib_symbolic_trace


class QuantLibRetracer(Retracer):
    def __init__(self):
        super(QuantLibRetracer, self).__init__('QuantLibRetracer', quantlib_symbolic_trace)
