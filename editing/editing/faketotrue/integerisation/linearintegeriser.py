from collections import OrderedDict
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, Callable, Union, NamedTuple, Type

from quantlib.editing.editing.editors import ModuleDescription, PathGraphDescription, PathGraphRewriter
from quantlib.editing.editing.editors import ApplicationPoint, ComposedEditor
from quantlib.editing.graphs import quantlib_fine_symbolic_trace
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel

from quantlib.algorithms.qmodules.qmodules.qlinears import _QLinear, QLinear, QConv1d, QConv2d, QConv3d
ConvNd = Union[nn.Conv1d, nn.Conv2d, nn.Conv3d]


class LinearOpIntegeriser(PathGraphRewriter):

    def __init__(self,
                 cls_name_prefix: str,
                 pgd:             PathGraphDescription,
                 from__qlinear:   Callable[[_QLinear], Union[nn.Linear, ConvNd]]):

        name = cls_name_prefix + 'Integeriser'
        super(LinearOpIntegeriser, self).__init__(name, pgd, quantlib_fine_symbolic_trace)

        self._from__qlinear = from__qlinear

        self.pattern.set_leakable_nodes(self.pattern.name_to_pattern_node()['eps_out'])

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        pass  # TODO: implement the check!

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap.core)
        node_eps_in  = name_to_match_node['eps_in']
        node_linear  = name_to_match_node['linear']
        node_eps_out = name_to_match_node['eps_out']

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap.core, data_gm=g)
        module_eps_in  = name_to_match_module['eps_in']
        module_linear  = name_to_match_module['linear']
        module_eps_out = name_to_match_module['eps_out']

        # create the integerised linear operation
        self._counter += 1

        new_module = self._from__qlinear(module_linear)
        iweight = torch.round(module_linear.qweight.data.clone().detach() / module_linear.scale.data.clone().detach())
        new_module.weight.data = iweight  # TODO: should I offload the responsibility of computing the true-quantised parameter array to `_QLinear`? Probably yes.
        new_target = '_'.join([self._editor_id.upper(), new_module.__class__.__name__.upper(), str(self._counter)])

        # add the requantised linear operation to the graph...
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node_eps_in):
            new_node = g.graph.call_module(new_target, args=(node_eps_in,))
        node_eps_out.replace_input_with(node_linear, new_node)

        module_eps_in.set_eps_out(torch.ones_like(module_eps_in._eps_out))  # TODO: must have the same shape as the input eps
        module_eps_out.set_eps_in(torch.ones_like(module_eps_out._eps_in))  # TODO: must have the same shape as the output eps

        # ...and delete the old operation
        g.delete_submodule(node_linear.target)
        g.graph.erase_node(node_linear)

        return g


class Pattern(NamedTuple):
    eps_in:  nn.Module
    linear:  nn.Module
    eps_out: nn.Module


class ExtendedPattern(NamedTuple):
    pattern:       Pattern
    from__qlinear: Callable[[_QLinear], Union[nn.Linear, ConvNd]]


class LinearOpIntegeriserFactory(object):

    def __init__(self):
        super(LinearOpIntegeriserFactory, self).__init__()

    @staticmethod
    def pgd_from_pattern(pattern: Pattern) -> PathGraphDescription:
        return (
            ModuleDescription(name='eps_in',  module=pattern.eps_in,  checkers=None),
            ModuleDescription(name='linear',  module=pattern.linear,  checkers=lambda m: m.bias is None),
            ModuleDescription(name='eps_out', module=pattern.eps_out, checkers=None),
        )

    @staticmethod
    def get_rewriter(cls_name_prefix: str, extpattern: ExtendedPattern) -> Type[LinearOpIntegeriser]:

        def __init__(self):
           LinearOpIntegeriser.__init__(self,
                                        cls_name_prefix,
                                        LinearOpIntegeriserFactory.pgd_from_pattern(extpattern.pattern),
                                        extpattern.from__qlinear)

        cls = type(cls_name_prefix + 'Integeriser', (LinearOpIntegeriser,), {'__init__': __init__})
        return cls


# programmatically create all the interising `Rewriter`s for linear operations (https://stackoverflow.com/a/15247892)

def from_qlinear(qm: QLinear) -> nn.Linear:
    module = nn.Linear(in_features=qm.in_features, out_features=qm.out_features, bias=qm.bias)
    return module


def from_qconvnd(qm: Union[QConv1d, QConv2d, QConv3d]) -> ConvNd:

    # identify base class
    if isinstance(qm, QConv1d):
        cls = nn.Conv1d
    elif isinstance(qm, QConv2d):
        cls = nn.Conv2d
    elif isinstance(qm, QConv3d):
        cls = nn.Conv3d
    else:
        raise TypeError

    # create replacement object
    module = cls(in_channels=qm.in_channels,
                 out_channels=qm.out_channels,
                 kernel_size=qm.kernel_size,
                 stride=qm.stride,
                 padding=qm.padding,
                 dilation=qm.dilation,
                 groups=qm.groups,
                 bias=qm.bias)

    return module


prefix_to_extpattern = OrderedDict([
    ('EpsInLinearEpsOut', ExtendedPattern(pattern=Pattern(eps_in=EpsTunnel(torch.Tensor([1.0])), linear=nn.Linear(in_features=1, out_features=1, bias=False), eps_out=EpsTunnel(torch.Tensor([1.0]))), from__qlinear=from_qlinear)),
    ('EpsInConv1dEpsOut', ExtendedPattern(pattern=Pattern(eps_in=EpsTunnel(torch.Tensor([1.0])), linear=nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False), eps_out=EpsTunnel(torch.Tensor([1.0]))), from__qlinear=from_qconvnd)),
    ('EpsInConv2dEpsOut', ExtendedPattern(pattern=Pattern(eps_in=EpsTunnel(torch.Tensor([1.0])), linear=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False), eps_out=EpsTunnel(torch.Tensor([1.0]))), from__qlinear=from_qconvnd)),
    ('EpsInConv3dEpsOut', ExtendedPattern(pattern=Pattern(eps_in=EpsTunnel(torch.Tensor([1.0])), linear=nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False), eps_out=EpsTunnel(torch.Tensor([1.0]))), from__qlinear=from_qconvnd)),
])


linopintegeriserclasses = {}
for cls_name_prefix, extpattern in prefix_to_extpattern.items():
    cls = LinearOpIntegeriserFactory.get_rewriter(cls_name_prefix, extpattern)  # define a new `LinOpIntegeriser` sub-class
    cls_name = cls.__name__
    globals()[cls_name] = cls                # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    linopintegeriserclasses[cls_name] = cls  # keep track of the new class so that we can later create the generic integeriser of linear operations

# `list(linopintegeriserclasses.keys())` now contains the name of all the newly-defined classes


class AllLinOpIntegeriser(ComposedEditor):

    def __init__(self):
        super(AllLinOpIntegeriser, self).__init__(list(map(lambda linopintegeriserclass: linopintegeriserclass(), linopintegeriserclasses.values())))
