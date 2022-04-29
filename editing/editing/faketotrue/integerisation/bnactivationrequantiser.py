from collections import OrderedDict
import itertools
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List, NamedTuple, Optional, Type

from quantlib.editing.editing.editors import ModuleDescription, PathGraphDescription, PathGraphRewriter
from quantlib.editing.editing.editors import ApplicationPoint, ComposedEditor
from quantlib.editing.graphs.nn.epstunnel import EpsTunnel
from quantlib.editing.graphs.nn.requant import Requantisation
from quantlib.editing.graphs import quantlib_fine_symbolic_trace


class Requantiser(PathGraphRewriter):

    def __init__(self,
                 cls_name_prefix: str,
                 pgd:             PathGraphDescription,
                 B:               int = 24):

        name = cls_name_prefix + 'Requantiser'
        super(Requantiser, self).__init__(name, pgd, quantlib_fine_symbolic_trace)

        self._D = torch.Tensor([2 ** B])

        self.pattern.set_leakable_nodes(self.pattern.name_to_pattern_node()['eps_out'])

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        pass  # TODO: implement the check!

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap.core)
        node_eps_in      = name_to_match_node['eps_in']
        node_bn          = name_to_match_node['bn'] if 'bn' in name_to_match_node.keys() else None
        node_qactivation = name_to_match_node['qactivation']
        node_eps_out     = name_to_match_node['eps_out']

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap.core, data_gm=g)
        module_eps_in      = name_to_match_module['eps_in']
        module_bn          = name_to_match_module['bn'] if 'bn' in name_to_match_module.keys() else None
        module_qactivation = name_to_match_module['qactivation']
        module_eps_out     = name_to_match_module['eps_out']

        assert ((node_bn is None) and (module_bn is None)) or (isinstance(node_bn, fx.Node) and isinstance(module_bn, nn.Module))

        # extract the parameters to create the requantiser
        eps_in  = module_eps_in._eps_out
        mi      = module_bn.running_mean if module_bn is not None else torch.zeros_like(eps_in)
        sigma   = torch.sqrt(module_bn.running_var + module_bn.eps) if module_bn is not None else torch.ones_like(eps_in)
        gamma   = module_bn.weight if module_bn is not None else torch.ones_like(eps_in)
        beta    = module_bn.bias if module_bn is not None else torch.zeros_like(eps_in)
        eps_out = module_eps_out._eps_in
        assert torch.all(eps_out == module_qactivation.scale)

        # compute the requantiser's parameters
        shape = node_qactivation.meta['tensor_meta'].shape
        broadcast_shape = tuple(1 if i != 1 else mi.numel() for i, _ in enumerate(range(0, len(shape))))
        mi    = mi.reshape(broadcast_shape)
        sigma = sigma.reshape(broadcast_shape)
        gamma = gamma.reshape(broadcast_shape)
        beta  = beta.reshape(broadcast_shape)

        gamma_int = torch.floor(self._D * (eps_in * gamma)             / (sigma * eps_out))
        beta_int  = torch.floor(self._D * (-mi * gamma + beta * sigma) / (sigma * eps_out))

        # create the requantiser
        self._counter += 1

        new_module = Requantisation(mul=gamma_int, add=beta_int, zero=module_qactivation.zero, n_levels=module_qactivation.n_levels, D=self._D)
        new_target = '_'.join([self._editor_id.upper(), new_module.__class__.__name__.upper(), str(self._counter)])

        # add the requantiser to the graph...
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node_eps_in):
            new_node = g.graph.call_module(new_target, args=(node_eps_in,))
        node_eps_out.replace_input_with(node_qactivation, new_node)

        module_eps_in.set_eps_out(torch.ones_like(module_eps_in._eps_out))  # TODO: must have the same shape as the input eps
        module_eps_out.set_eps_in(torch.ones_like(module_eps_out._eps_in))  # TODO: must have the same shape as the output eps

        # ...and delete the old construct
        g.delete_submodule(node_qactivation.target)
        g.graph.erase_node(node_qactivation)  # since `node_qactivation` is a user of `node_bn`, we must delete it first
        if node_bn is not None:
            g.delete_submodule(node_bn.target)
            g.graph.erase_node(node_bn)

        return g


class Pattern(NamedTuple):
    eps_in:      nn.Module
    bn:          Optional[nn.Module]
    qactivation: nn.Module
    eps_out:     nn.Module


class RequantiserFactory(object):

    def __init__(self):
        super(RequantiserFactory, self).__init__()

    @staticmethod
    def pgd_from_pattern(pattern: Pattern) -> PathGraphDescription:
        module_descriptions = []
        module_descriptions += [ModuleDescription(name='eps_in',      module=pattern.eps_in,      checkers=None)]
        module_descriptions += [ModuleDescription(name='bn',          module=pattern.bn,          checkers=None)] if pattern.bn is not None else []
        module_descriptions += [ModuleDescription(name='qactivation', module=pattern.qactivation, checkers=None)]  # TODO: checking whether the module `is_quantised` will fail while validating the `PathGraphDescription` since I am not really using `_QModule`s in the patterns
        module_descriptions += [ModuleDescription(name='eps_out',     module=pattern.eps_out,     checkers=None)]
        return tuple(module_descriptions)

    @staticmethod
    def get_rewriter(cls_name_prefix: str, pattern: Pattern) -> Type[Requantiser]:

        def __init__(self,
                     B: int = 24):
            Requantiser.__init__(self,
                                 cls_name_prefix,
                                 RequantiserFactory.pgd_from_pattern(pattern),
                                 B=B)

        cls = type(cls_name_prefix + 'Requantiser', (Requantiser,), {'__init__': __init__})
        return cls


# programmatically create all the requantising `Rewriter`s (https://stackoverflow.com/a/15247892)
eps_in_modules = OrderedDict([
    ('EpsIn', EpsTunnel(eps=torch.Tensor([1.0]))),
])

bn_modules = OrderedDict([
    ('',     None),
    ('BN1d', nn.BatchNorm1d(num_features=1)),
    ('BN2d', nn.BatchNorm2d(num_features=1)),
    ('BN3d', nn.BatchNorm3d(num_features=1)),
])

qactivation_modules = OrderedDict([
    ('QIdentity',  nn.Identity()),
    ('QReLU',      nn.ReLU(inplace=True)),
    ('QReLU6',     nn.ReLU6(inplace=True)),
    ('QLeakyReLU', nn.LeakyReLU(inplace=True)),
])

eps_out_modules = OrderedDict([
    ('EpsOut', EpsTunnel(eps=torch.Tensor([1.0]))),
])

modules = OrderedDict(**eps_in_modules,
                      **bn_modules,
                      **qactivation_modules,
                      **eps_out_modules)

patterns = list(itertools.product(eps_in_modules, bn_modules, qactivation_modules, eps_out_modules))
prefix_to_pattern = OrderedDict(list(map(lambda t: (''.join([t[0], t[1], t[2], t[3]]), Pattern(eps_in=modules[t[0]], bn=modules[t[1]], qactivation=modules[t[2]], eps_out=modules[t[3]])), patterns)))

requantiserclasses = {}
for cls_name_prefix, pattern in prefix_to_pattern.items():
    cls = RequantiserFactory.get_rewriter(cls_name_prefix, pattern)  # define a new `Requantiser` sub-class
    cls_name = cls.__name__
    globals()[cls_name] = cls           # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    requantiserclasses[cls_name] = cls  # keep track of the new class so that we can later create the generic requantiser

# `list(requantiserclasses.keys())` now contains the name of all the newly-defined classes


class AllRequantiser(ComposedEditor):

    def __init__(self, B: int = 24):
        super(AllRequantiser, self).__init__(list(map(lambda requantiserclass: requantiserclass(B=B), requantiserclasses.values())))
