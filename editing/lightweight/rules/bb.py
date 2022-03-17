from typing import Union

from torch import nn
from functools import partial

from quantlib.editing.lightweight.rules import LightweightRule

from quantlib.editing.lightweight.rules.filters import Filter, VariadicOrFilter, NameFilter, TypeFilter

class ReplaceConvLinearBBRule(LightweightRule):
    @staticmethod
    def replace_bb_conv_linear(module : Union[nn.Conv2d, nn.Linear], **kwargs):
        # pretty ugly hack to avoid circular import
        from quantlib.algorithms.bb.bb_ops import BBConv2d, BBLinear
        if isinstance(module, nn.Conv2d):
            return BBConv2d.from_conv2d(module, **kwargs)
        elif isinstance(module, nn.Linear):
            return BBLinear.from_linear(module, **kwargs)
        else:
            raise TypeError(f"Incompatible module of type {module.__class__.__name__} passed to replace_bb_conv_linear!")

    def __init__(self,
                 filter_ : Filter,
                 **kwargs):
        replacement_fun = partial(self.replace_bb_conv_linear, **kwargs)
        super(ReplaceConvLinearBBRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)


class ReplaceActBBRule(LightweightRule):
    @staticmethod
    def replace_bb_act(module : nn.Module,
                       **kwargs):
        from quantlib.algorithms.bb.bb_ops import BBAct
        if 'act_kind' not in kwargs.keys():
            if isinstance(module, nn.ReLU6):
                act_kind = 'relu6'
            elif isinstance(module, nn.LeakyReLU):
                act_kind = 'leaky_relu'
                if 'leaky' not in kwargs:
                    kwargs['leaky'] = module.negative_slope
            else: # default activation is ReLU
                act_kind = 'relu'

            kwargs['act_kind'] = act_kind
        return BBAct(**kwargs)

    def __init__(self,
                 filter_ : Filter,
                 **kwargs):
        replacement_fun = partial(self.replace_bb_act, **kwargs)
        super(ReplaceActBBRule, self).__init__(filter_=filter_, replacement_fun=replacement_fun)
