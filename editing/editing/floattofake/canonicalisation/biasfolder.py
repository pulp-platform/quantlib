from collections import OrderedDict
import torch.nn as nn
import torch.fx as fx
from typing import NamedTuple, List, Type

from quantlib.editing.editing.editors.pathgraphs import ModuleDescription, PathGraphDescription, PathGraphRewriter
from quantlib.editing.editing.editors.base import ApplicationPoint, ComposedEditor
from quantlib.editing.graphs import quantlib_fine_symbolic_trace


class BiasFolder(PathGraphRewriter):
    """A template class for all bias-folding rewriting rules."""

    def __init__(self, cls_name_prefix: str, pgd: PathGraphDescription):
        name = cls_name_prefix + self.__class__.__name__
        super(BiasFolder, self).__init__(name, pgd, quantlib_fine_symbolic_trace)

    def _check_aps_independence(self, aps: List[ApplicationPoint]) -> None:
        pass  # TODO: implement the check!

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:

        # get handles on matched `nn.Module`s
        name_to_match_module = self.pattern.name_to_match_module(nodes_map=ap.core, data_gm=g)
        module_linear        = name_to_match_module['linear']
        module_bn            = name_to_match_module['bn']

        # modify matched `nn.Module`s in-place
        bias                         = module_linear.bias.data.detach().clone()
        module_linear.bias           = None
        module_bn.running_mean.data -= bias

        return g


class Pattern(NamedTuple):
    linear: nn.Module
    bn:     nn.Module


class BiasFolderFactory(object):
    """A class automating the creation of bias-folding ``Rewriter``s."""

    def __init__(self):
        super(BiasFolderFactory, self).__init__()

    @staticmethod
    def pgd_from_pattern(pattern: Pattern) -> PathGraphDescription:
        return (
            ModuleDescription(name='linear', module=pattern.linear, checkers=lambda m: m.bias is not None),
            ModuleDescription(name='bn',     module=pattern.bn,     checkers=None),
        )

    @staticmethod
    def get_rewriter(cls_name_prefix: str, pattern: Pattern) -> Type[BiasFolder]:

        def __init__(self):
            BiasFolder.__init__(self, cls_name_prefix, BiasFolderFactory.pgd_from_pattern(pattern))

        cls = type(cls_name_prefix + 'BiasFolder', (BiasFolder,), {'__init__': __init__})
        return cls


# programmatically create all the bias folding `Rewriter`s (https://stackoverflow.com/a/15247892)
prefix_to_pattern = OrderedDict([
    ('LinearBN1d', Pattern(linear=nn.Linear(in_features=1, out_features=1, bias=True),                                     bn=nn.BatchNorm1d(num_features=1),)),
    ('Conv1dBN1d', Pattern(linear=nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True), bn=nn.BatchNorm1d(num_features=1),)),
    ('Conv2dBN2d', Pattern(linear=nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True), bn=nn.BatchNorm2d(num_features=1),)),
    ('Conv3dBN3d', Pattern(linear=nn.Conv3d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True), bn=nn.BatchNorm3d(num_features=1),)),
])

biasfolderclasses = OrderedDict([])
for cls_name_prefix, pattern in prefix_to_pattern.items():
    cls = BiasFolderFactory.get_rewriter(cls_name_prefix, pattern)  # define a new `BiasFolder` sub-class
    cls_name = cls.__name__
    globals()[cls_name] = cls                         # add the new class to the module's namespace, so that it can be exported to QuantLib's namespace
    biasfolderclasses[cls_name] = cls                 # keep track of the new class so that we can later create the generic bias folder

# `list(biasfolderclasses.keys())` now contains the name of all the newly-defined classes


class AllLinearBNBiasFolder(ComposedEditor):
    def __init__(self):
        super(AllLinearBNBiasFolder, self).__init__(list(map(lambda biasfolderclass: biasfolderclass(), biasfolderclasses.values())))
