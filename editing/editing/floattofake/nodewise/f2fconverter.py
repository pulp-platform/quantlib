import copy
import torch.nn as nn
import torch.fx as fx
from typing import List, Tuple, Dict, NamedTuple, Type, Any

from .f2fspecification import F2FSpecType, resolve_f2fspec, F2FPartition
from quantlib.editing.editing.editors.editors import ApplicationPoint, Rewriter
from quantlib.algorithms.qbase import QRange, QGranularity, QHParamsInitStrategy
from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule


class NodeConversionInfo(NamedTuple):
    target:               str                   # the qualified name of the target `nn.Module`
    fpm:                  nn.Module             # the full-precision `nn.Module` object to be replaced
    qmclass:              Type[_QModule]        # the sub-class of `_QModule` such as `PACTConv2d` or `PACTReLU` which should be used to replace the target `nn.Module`
    qrange:               QRange                # the target integer range of weights (for linear operations) or features (for activations)
    qgranularity:         QGranularity          # the sub-tensor structure where each sub-tensor's quantiser has a different scale with respect to its sibling quantisers
    qhparamsinitstrategy: QHParamsInitStrategy  # the initialisation strategy for the quantisers
    qmclasskwargs:        Dict[str, Any]        # remaining arguments for the target `_QModule`'s constructor


class F2FConverter(Rewriter):

    def __init__(self,
                 f2fspec: F2FSpecType):

        name = 'F2FConverter'
        super(F2FConverter, self).__init__(name)

        self._f2fspec: List[F2FPartition] = resolve_f2fspec(f2fspec)

    @staticmethod
    def _get_nodeconversioninfos(f2fpartition: F2FPartition, name_to_module: Dict[str, nn.Module]) -> List[NodeConversionInfo]:

        partition_name_to_module = f2fpartition.modulefilter(name_to_module)

        nodeconversioninfos = []

        for target, fpm in partition_name_to_module.items():

            qmclass              = f2fpartition.ptqqatinfo.mapping[type(fpm)]
            qrange               = copy.copy(f2fpartition.qrange)
            qgranularity         = copy.copy(f2fpartition.qgranularity)
            qhparamsinitstrategy = copy.copy(f2fpartition.qhparamsinitstrategy)
            qmclasskwargs        = copy.copy(f2fpartition.ptqqatinfo.kwargs)

            nci = NodeConversionInfo(target=target,
                                     fpm=fpm,
                                     qmclass=qmclass,
                                     qrange=qrange,
                                     qgranularity=qgranularity,
                                     qhparamsinitstrategy=qhparamsinitstrategy,
                                     qmclasskwargs=qmclasskwargs)

            nodeconversioninfos.append(nci)

        return nodeconversioninfos

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:

        name_to_module = dict(g.named_modules())

        nodeconversioninfos = []
        for f2fpartition in self._f2fspec:
            nodeconversioninfos.extend(F2FConverter._get_nodeconversioninfos(f2fpartition, name_to_module))

        return nodeconversioninfos

    def _check_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:
        # TODO: before performing the checks, we need either to extend the definition of application point cores to `NodeConversionInfo`s, or to restructure the flow of the node-wise editing
        pass

    @staticmethod
    def split_path_to_target(target: str) -> Tuple[str, str]:
        *ancestors, child = target.rsplit('.')
        path_to_parent = '.'.join(ancestors) if len(ancestors) > 0 else ''
        return path_to_parent, child

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        name_to_module = dict(g.named_modules())

        new_module = ap.qmclass.from_fp_module(ap.fpm, ap.qrange, ap.qgranularity, ap.qhparamsinitstrategy, **ap.qmclasskwargs)
        name_to_module[ap.target] = new_module

        path_to_parent, child = F2FConverter.split_path_to_target(ap.target)
        setattr(name_to_module[path_to_parent], child, new_module)  # https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L44

        return g
