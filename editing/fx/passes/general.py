from torch import nn, fx
from .pass_base import SequentialPass, ModifySequentialPatternPass

__all__ = ['MergeConvBNPass']

def merge_conv_bn_fun(ml : list):
    assert len(ml) == 2, "List passed to merge_conv_bn_fun should have length 2"
    conv_module = [m for m in ml if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d))][0]
    bn_module = [m for m in ml if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))][0]
    if conv_module.bias is None:
        return
    bias_data = conv_module.bias.data.clone().detach()
    conv_module.bias = None
    bn_module.running_mean.data -= bias_data

class MergeConvBNPass(SequentialPass):
    def __init__(self, trace : callable = fx.symbolic_trace):
        passes = []
        for conv, bn, dim in [(nn.Conv1d, nn.BatchNorm1d, "1D"), (nn.Conv2d, nn.BatchNorm2d, "2D"), (nn.Conv3d, nn.BatchNorm3d, "3D")]:
            pattern = nn.Sequential(conv(1,1,1), bn(1))
            passes.append(ModifySequentialPatternPass(pattern, trace, merge_conv_bn_fun, f"_MERGE_CONV_BN_{dim}"))
        super(MergeConvBNPass, self).__init__(*passes, name_prefix="_MERGE_CONV_BN_PASS")
