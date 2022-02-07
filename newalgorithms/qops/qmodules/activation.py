import torch
import torch.nn as nn

from quantlib.newalgorithms.qbase import QRangeSpecType, resolve_qrangespec
from quantlib.newalgorithms.qbase import init_qhparams
from quantlib.newalgorithms.qbase import MinMaxMeanVarObserver
from quantlib.newutils import quantlib_err_header


class QActivation(nn.Module):

    def __init__(self,
                 qrangespec: QRangeSpecType,
                 qinitstrategy: str,
                 nonqmodule: nn.Module = nn.Identity(),
                 *args,
                 **kwargs):

        super().__init__()

        # as long as the module is not fake-quantised (i.e., the quantiser hyper-parameters are not fully resolved), this object will operate as a standard module
        self._nonqmodule = nonqmodule

        # initialise the hyper-parameters describing the shape of the quantisers
        self._qrange        = resolve_qrangespec(qrangespec)
        self._qinitstrategy = qinitstrategy

        zero, n_levels, step, scale = init_qhparams(self._qrange, self._granularity)
        self._zero     = self.register_buffer('zero',     zero)
        self._n_levels = self.register_buffer('n_levels', n_levels)
        self._step     = self.register_buffer('step',     step)
        self._scale    = self.register_buffer('scale',    scale)

        # prepare the machinery to finalise the initialisation of the quantiser hyper-parameters
        self._observer = None
        self._is_observing = self.register_buffer('is_observing', torch.Tensor([False]))
        self._is_fakequant = self.register_buffer('is_fakequant', torch.Tensor([False]))

        # this will be instantiated by the specific PTQ/QAT algorithm
        self._qautogradop = None
        self._init_qautogradop(*args, **kwargs)

    def _init_qautogradop(self, *args, **kwargs):
        raise NotImplementedError

    def start_observing(self):
        self._observer = MinMaxMeanVarObserver()
        self._is_observing |= True

    def stop_observing(self):
        self._is_observing &= False
        self._observer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self._is_fakequant:
            if self._is_observing:
                raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "is in an inconsistent state: it can not be observing while fake-quantised.")
            else:
                x = self._qautogradop(x)

        else:
            if self._is_observing:
                with torch.no_grad():
                    self._observer.update(x)
            x = self._nonqmodule(x)

        return x
