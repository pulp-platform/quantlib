from collections import OrderedDict
import itertools
import torch.nn as nn
import torch.fx as fx
from typing import NamedTuple, List, Type, Dict, Any

from quantlib.editing.editing.editors import ModuleDescription, PathGraphDescription, PathGraphRewriter
from quantlib.editing.editing.editors import ApplicationPoint, ComposedEditor
from quantlib.editing.graphs import quantlib_fine_symbolic_trace
from quantlib.algorithms.qalgorithms import ptqqat_index
from quantlib.algorithms.qbase import QRangeSpecType, QGranularitySpecType, QHParamsInitStrategySpecType
from quantlib.algorithms.qmodules.qmodules.qactivations import _QActivation


class QuantiserInterposer(PathGraphRewriter):

    def __init__(self,
                 cls_name_prefix:          str,
                 pgd:                      PathGraphDescription,
                 algorithm:                str,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        name = cls_name_prefix + 'QuantiserInterposer'
        super(QuantiserInterposer, self).__init__(name, pgd, quantlib_fine_symbolic_trace)

        self._quantiser_cls            = ptqqat_index.register[algorithm][nn.Identity]
        self._qrangespec               = qrangespec
        self._qgranularityspec         = qgranularityspec
        self._qhparamsinitstrategyspec = qhparamsinitstrategyspec

        self.pattern.set_leakable_nodes(self.pattern.name_to_pattern_node()['linear_pre'])

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        pass  # TODO: implement the check!

    def _create_quantiser(self) -> _QActivation:
        return self._quantiser_cls(qrangespec=self._qrangespec,
                                   qgranularityspec=self._qgranularityspec,
                                   qhparamsinitstrategyspec=self._qhparamsinitstrategyspec)

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        # get handles on matched `fx.Node`s
        name_to_match_node = self.pattern.name_to_match_node(nodes_map=ap.core)
        node_pre  = name_to_match_node['linear_pre']
        node_post = name_to_match_node['linear_post']

        # create the quantiser
        self._counter += 1

        new_module = self._create_quantiser()
        new_target = '_'.join([self._editor_id.upper(), new_module.__class__.__name__.upper(), str(self._counter)])

        # add the quantiser to the graph (interposing it between the two linear nodes)
        g.add_submodule(new_target, new_module)
        with g.graph.inserting_after(node_pre):
            new_node = g.graph.call_module(new_target, args=(node_pre,))
        downstream_nodes = list(u for u in node_pre.users if u != new_node)  # TODO: this is ugly, but at the same time `n.replace_all_uses_with` leads to a "circular dependency" where `new_node` becomes an input to itself...
        assert node_post in downstream_nodes
        for u in downstream_nodes:
            u.replace_input_with(node_pre, new_node)

        return g


class Pattern(NamedTuple):
    linear_pre:  nn.Module
    linear_post: nn.Module


class QuantiserInterposerFactory(object):

    def __init__(self):
        super(QuantiserInterposerFactory, self).__init__()

    @staticmethod
    def pgd_from_pattern(pattern: Pattern) -> PathGraphDescription:
        return (
            ModuleDescription(name='linear_pre',  module=pattern.linear_pre,  checkers=None),
            ModuleDescription(name='linear_post', module=pattern.linear_post, checkers=None),
        )

    @staticmethod
    def get_rewriter(cls_name_prefix: str, pattern: Pattern) -> Type[QuantiserInterposer]:

        def __init__(self,
                     algorithm:                str,
                     qrangespec:               QRangeSpecType,
                     qgranularityspec:         QGranularitySpecType,
                     qhparamsinitstrategyspec: QHParamsInitStrategySpecType):
            QuantiserInterposer.__init__(self,
                                         cls_name_prefix,
                                         QuantiserInterposerFactory.pgd_from_pattern(pattern),
                                         algorithm,
                                         qrangespec,
                                         qgranularityspec,
                                         qhparamsinitstrategyspec)

        cls = type(cls_name_prefix + 'QuantiserInterposer', (QuantiserInterposer,), {'__init__': __init__})
        return cls


# programmatically create all the quantiser interposing `Rewriter`s (https://stackoverflow.com/a/15247892)
class ModuleInfo(NamedTuple):
    cls:    Type[nn.Module]
    kwargs: Dict[str, Any]


moduleinfos = OrderedDict([
    ('Linear', ModuleInfo(cls=nn.Linear, kwargs={'in_features': 1, 'out_features': 1, 'bias': False})),
    ('Conv1d', ModuleInfo(cls=nn.Conv1d, kwargs={'in_channels': 1, 'out_channels': 1, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': False})),
    ('Conv2d', ModuleInfo(cls=nn.Conv2d, kwargs={'in_channels': 1, 'out_channels': 1, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': False})),
    ('Conv3d', ModuleInfo(cls=nn.Conv3d, kwargs={'in_channels': 1, 'out_channels': 1, 'kernel_size': 1, 'stride': 1, 'padding': 0, 'bias': False})),
    ('BN1d',   ModuleInfo(cls=nn.BatchNorm1d, kwargs={'num_features': 1})),
    ('BN2d',   ModuleInfo(cls=nn.BatchNorm2d, kwargs={'num_features': 1})),
    ('BN3d',   ModuleInfo(cls=nn.BatchNorm3d, kwargs={'num_features': 1})),
])

problematic_successors = OrderedDict([
    ('Linear', ('Linear',)),
    ('Conv1d', ('Conv1d',)),
    ('Conv2d', ('Conv2d',)),
    ('Conv3d', ('Conv3d',)),
    ('BN1d',   ('Linear', 'Conv1d',)),
    ('BN2d',   ('Conv2d',)),
    ('BN3d',   ('Conv3d',))
])

problematic_patterns = list(itertools.chain(*[itertools.product((k,), v) for k, v in problematic_successors.items()]))
prefix_to_pattern = OrderedDict(list(map(lambda t: (t[0] + t[1], Pattern(linear_pre=moduleinfos[t[0]].cls(**moduleinfos[t[0]].kwargs), linear_post=moduleinfos[t[1]].cls(**moduleinfos[t[1]].kwargs))), problematic_patterns)))

quantiserinterposerclasses = {}
for cls_name_prefix, pattern in prefix_to_pattern.items():
    cls = QuantiserInterposerFactory.get_rewriter(cls_name_prefix, pattern)  # define a new `QuantiserInterposer` sub-class
    cls_name = cls.__name__
    globals()[cls_name] = cls                   # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    quantiserinterposerclasses[cls_name] = cls  # keep track of the new class so that we can later create the generic quantiser interposer

# `list(quantiserinterposerclasses.keys())` now contains the name of all the newly-defined classes


class AllQuantiserInterposer(ComposedEditor):

    def __init__(self,
                 algorithm:                str,
                 qrangespec:               QRangeSpecType,
                 qgranularityspec:         QGranularitySpecType,
                 qhparamsinitstrategyspec: QHParamsInitStrategySpecType):

        super(AllQuantiserInterposer, self).__init__(list(map(lambda quantiserinterposerclass: quantiserinterposerclass(algorithm, qrangespec, qgranularityspec, qhparamsinitstrategyspec), quantiserinterposerclasses.values())))
