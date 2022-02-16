import copy
import torch.nn as nn
import torch.fx as fx
from typing import List, Dict, Optional, NamedTuple, Type, Any, Tuple

from .f2fspecification import F2FSpecType, resolve_f2fspec, F2FPartition
from ..gebase import GraphRewriter, ApplicationPoint
from quantlib.newalgorithms.qbase import QRange, QGranularity, QHParamsInitStrategy
from quantlib.newalgorithms.qmodules.qmodules.qmodules import _QModule


class NodeConversionInfo(NamedTuple):
    target:               str
    fpm:                  nn.Module
    qmclass:              Type[_QModule]
    qrange:               QRange
    qgranularity:         QGranularity
    qhparamsinitstrategy: QHParamsInitStrategy
    qmclasskwargs:        Dict[str, Any]


class F2FConverter(GraphRewriter):

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

    def find_application_points(self, data_gm: fx.GraphModule) -> List[ApplicationPoint]:

        name_to_module = dict(data_gm.named_modules())

        nodeconversioninfos = []
        for f2fpartition in self._f2fspec:
            nodeconversioninfos.extend(F2FConverter._get_nodeconversioninfos(f2fpartition, name_to_module))

        # TODO: check that the same target is not specified more than once

        return nodeconversioninfos

    @staticmethod
    def split_path_to_target(target: str) -> Tuple[str, str]:
        *ancestors, child = target.rsplit('.')
        path_to_parent = '.'.join(ancestors) if len(ancestors) > 0 else ''
        return path_to_parent, child

    def _apply(self, data_gm: fx.GraphModule, ap: ApplicationPoint):

        name_to_module = dict(data_gm.named_modules())

        new_module = ap.qmclass.from_fp_module(ap.fpm, ap.qrange, ap.qgranularity, ap.qhparamsinitstrategy, **ap.qmclasskwargs)
        name_to_module[ap.target] = new_module

        path_to_parent, child = F2FConverter.split_path_to_target(ap.target)
        setattr(name_to_module[path_to_parent], child, new_module)  # https://github.com/pytorch/pytorch/blob/40cbf342d3c000712da92cfafeaca651b3e0bd3e/torch/fx/experimental/optimization.py#L44

    def apply(self, data_gm: fx.GraphModule, ap: Optional[ApplicationPoint] = None):

        for ap in self.find_application_points(data_gm):
            self._apply(data_gm, ap)
