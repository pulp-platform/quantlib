from torch import nn, fx
from torch.nn import functional as F
from .pass_base import SequentialPass, ModifySequentialPatternPass, ModularizePass

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

class ModularizeActivationsPass(ModularizePass):

    inplace_act_functions = (F.threshold_,
                             F.relu_,
                             F.hardtanh_,
                             F.elu_,
                             F.leaky_relu_,
                             F.rrelu_)

    act_function_to_module = {F.threshold : nn.Threshold,
                              F.threshold_ : nn.Threshold,
                              F.relu : nn.ReLU,
                              F.relu_ : nn.ReLU,
                              F.hardtanh : nn.Hardtanh,
                              F.hardswish : nn.Hardswish,
                              F.relu6 : nn.ReLU6,
                              F.elu : nn.ELU,
                              F.elu_ : nn.ELU,
                              F.selu : nn.SELU,
                              F.celu : nn.CELU,
                              F.leaky_relu : nn.LeakyReLU,
                              F.leaky_relu_ : nn.LeakyReLU,
                              F.prelu : nn.PReLU,
                              F.rrelu : nn.RReLU,
                              F.rrelu_ : nn.RReLU,
                              F.glu : nn.GLU,
                              F.gelu : nn.GELU,
                              F.logsigmoid : nn.LogSigmoid,
                              F.hardshrink : nn.Hardshrink,
                              F.tanhshrink : nn.Tanhshrink,
                              F.softsign : nn.Softsign,
                              F.softplus : nn.Softplus,
                              F.softmin : nn.Softmin,
                              F.softmax : nn.Softmax,
                              F.softshrink : nn.Softshrink,
                              #F.log_softmax : nn.LogSoftmax # interfaces don't
                              #conform as they should for logSoftmax...
                              F.tanh : nn.Tanh,
                              F.sigmoid : nn.Sigmoid,
                              F.hardsigmoid : nn.Hardsigmoid,
                              F.silu : nn.SiLU,
                              F.mish : nn.Mish}


    @staticmethod
    def act_node_to_module(node):
        module_inst_args = node.args[1:]
        module_inst_kwargs = {k:v for k,v in node.kwargs.items() if k != 'input'}
        if node.target in inplace_act_functions:
            module_inst_kwargs['inplace'] = True
        module_call_args = node.args[0:1]
        module_call_kwargs = {k:v for k,v in node.kwargs.items() if k == 'input'}
        module_class = act_function_to_module[node.target]
        return (module_class(*module_inst_args, **module_inst_kwargs), module_call_args, module_call_kwargs)

    def __init__(self):
        super(ModularizeActivationsPass, self).__init__(op='call_function', target=tuple(k for k in act_function_to_module.keys()), name="MODULARIZE_ACTIVATIONS_PASS")
