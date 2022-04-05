from copy import deepcopy
import torch
from torch import nn, fx
from torch.fx.subgraph_rewriter import Match
from torch.nn.modules.utils import _single, _pair, _triple

from quantlib.algorithms.pact import RequantShift
from quantlib.editing.fx.passes import SequentialPass, ReplaceSequentialPatternPass, ShapePropPass
from quantlib.editing.fx.util import get_ordered_active_nodes, module_of_node
from quantlib.editing.fx.passes.pact import PACT_symbolic_trace


class AvgPoolWrap(nn.Sequential):
    # the avg pool wrapper is identical to nn.Sequential, we just need it to
    # distinguish wrapped AvgPool nodes
    def __init__(self, *args):
        super(AvgPoolWrap, self).__init__(*args)

class AlignAvgPoolPass(SequentialPass):
    GLOBAL_AVGPOOL_CLS = (nn.AdaptiveAvgPool1d,
                       nn.AdaptiveAvgPool2d,
                       nn.AdaptiveAvgPool3d)
    AVGPOOL_CLS = (nn.AvgPool1d,
                   nn.AvgPool2d,
                   nn.AvgPool3d)
    # the purpose of this pass is to align the 'mul' attribute of a
    # requantShift node to the total kernel size
    @staticmethod
    def align_and_wrap_avg_pool(gm : fx.GraphModule, match : Match):
        n = get_ordered_active_nodes(match)
        m = [module_of_node(gm, node) for node in n]
        if isinstance(m[0], AlignAvgPoolPass.GLOBAL_AVGPOOL_CLS):
            shape_in = n[0].all_input_nodes[0].meta['tensor_meta'].shape
            shape_out = n[0].meta['tensor_meta'].shape
            # strip batch and channel dimensions
            shape_in = shape_in[2:]
            shape_out = shape_out[2:]
            ks_tot = 1
            for i, (in_d, out_d) in enumerate(zip(shape_in, shape_out)):
                assert in_d % out_d == 0, "AlignAvgPoolPass: Non-integer kernel size in dimension {i} - in_dim {in_d}, out_dim {out_d}!"
                ks_tot *= (in_d//out_d)
        elif isinstance(m[0], AlignAvgPoolPass.AVGPOOL_CLS):
            dim = int(m[0].__class__.__name__[-2])
            if dim == 1:
                ks = _single(m[0].kernel_size)
            elif dim == 2:
                ks = _pair(m[0].kernel_size)
            else:
                ks = _triple(m[0].kernel_size)
            ks_tot = 1
            for k in ks:
                ks_tot *= k
        new_avgpool = deepcopy(m[0])
        new_rqs = deepcopy(m[1])
        # this is why we do this pass in the first place: for bit-true
        # inference, the `mul` attribute must be aligned to the total kernel
        # size
        new_rqs.mul.data = torch.round(new_rqs.mul.data/ks_tot)*ks_tot
        return AvgPoolWrap(new_avgpool, new_rqs)

    def __init__(self):
        passes = []
        for ap_cls in self.GLOBAL_AVGPOOL_CLS + self.AVGPOOL_CLS:
            nd = int(ap_cls.__name__[-2])
            name = f"AVGPOOL{nd}D_DORY"
            if 'Global' in ap_cls.__name__:
                name = "GLOBAL_" + name
            pattern = nn.Sequential(ap_cls(1), RequantShift(torch.zeros([]), torch.zeros([]), 256))
            passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, self.align_and_wrap_avg_pool, name))

        super(AlignAvgPoolPass, self).__init__(*passes)


class DORYHarmonizePass(SequentialPass):
    def __init__(self, in_shape):
        passes = []
        passes.append(ShapePropPass(in_shape))
        passes.append(AlignAvgPoolPass())
        super(DORYHarmonizePass, self).__init__(*passes)


