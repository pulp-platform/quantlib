import torch
import torch.nn as nn
from typing import Tuple

from quantlib.newalgorithms.qalgorithms import ptqqat_index
from quantlib.newalgorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType

from quantlib.newalgorithms.qmodules.qmodules.qmodules import _QModule


class AddTreeHarmoniser(nn.Module):

    def __init__(self,
                 ap,  #: OpTree,
                 algorithm: str,
                 qrangespec: QRangeSpecType,
                 qgranularityspec: QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                 force_output_scale: bool = False):
        # TODO: add `kwargs` for the constructor of the harmonising ``_QModule``s

        super(AddTreeHarmoniser, self).__init__()

        self._input_qmodules = torch.nn.ModuleList(list(map(lambda n: AddTreeHarmoniser.get_qmodule(algorithm, qrangespec, qgranularityspec, qhparamsinitstrategyspec), ap.inbound_frontier)))
        self._output_qmodule = AddTreeHarmoniser.get_qmodule(algorithm, qrangespec, qgranularityspec, qhparamsinitstrategyspec)

        self._force_output_scale = force_output_scale

    @staticmethod
    def get_qmodule(algorithm: str,
                    qrangespec: QRangeSpecType,
                    qgranularityspec: QGranularitySpecType,
                    qhparamsinitstrategyspec: QHParamsInitStrategySpecType,
                    **kwargs) -> _QModule:

        qclass = ptqqat_index.register[algorithm][nn.Identity]
        qmodule = qclass(qrangespec=qrangespec,
                         qgranularityspec=qgranularityspec,
                         qhparamsinitstrategyspec=qhparamsinitstrategyspec,
                         **kwargs)

        return qmodule

    # TODO:
    # All the ``_QModule``s in a ``Harmoniser`` should be kept synchronised.
    # This synchronisation involves the attributes that are not functional to
    # the harmonisation itself (e.g., ``training`` and ``_is_quantised`` flags).

    @property
    def is_training(self) -> bool:
        are_input_qmodules_training = all(map(lambda qm: qm.training, self._input_qmodules))
        is_output_qmodule_training = self._output_qmodule.training
        return are_input_qmodules_training and is_output_qmodule_training

    @property
    def is_quantised(self) -> bool:
        are_input_qmodules_quantised = all(map(lambda qm: qm._is_quantised, self._input_qmodules))
        is_output_qmodule_quantised = self._output_qmodule._is_quantised
        return are_input_qmodules_quantised and is_output_qmodule_quantised

    def start_observing(self) -> None:
        for qm in self._input_qmodules:
            qm.start_observing()
        self._output_qmodule.start_observing()

    def stop_observing(self) -> None:
        for qm in self._input_qmodules:
            qm.stop_observing()
        self._output_qmodule.stop_observing()

    def _harmonise(self) -> None:

        if self._force_output_scale:
            ref_module = self._output_qmodule
        else:
            raise NotImplementedError

        scale = ref_module.scale.detach().clone()  # https://discuss.pytorch.org/t/difference-between-detach-clone-and-clone-detach/34173/2
        clip_lo = ref_module.clip_lo.detach().clone()
        clip_hi = ref_module.clip_hi.detach().clone()

        for qm in self._input_qmodules:
            qm.scale.data.copy_(scale)
            qm.clip_lo.data.copy_(clip_lo)
            qm.clip_hi.data.copy_(clip_hi)

    def forward(self, *args: Tuple[torch.Tensor]) -> torch.Tensor:

        if self.is_training and self.is_quantised:
            self._harmonise()

        sum_ = self._input_qmodules[0](args[0])
        for i, (qm, x) in enumerate(zip(self._input_qmodules[1:], args[1:])):
            sum_ = sum_ + qm(x)

        return self._output_qmodule(sum_)
