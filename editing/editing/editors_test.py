import unittest
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List

from .editors.editors import Annotator
from .editors.editors import ApplicationPoint, Rewriter
from .editors.editors import ComposedEditor
import quantlib.editing.graphs as qeg
from quantlib.utils import quantlib_err_header


class MLP(nn.Module):
    """A simple multi-layer perceptron to verify editing and debugging flows."""

    def __init__(self):

        super(MLP, self).__init__()

        self._layer1 = self._make_layer(64, 64)
        self._layer2 = self._make_layer(64, 64)
        self._classifier = nn.Linear(in_features=64, out_features=10, bias=False)

        self._initialise_parameters()

    def _make_layer(self, in_features: int, out_features: int) -> nn.Sequential:
        modules  = []
        modules += [nn.Linear(in_features=in_features, out_features=out_features, bias=False)]
        modules += [nn.BatchNorm1d(num_features=out_features)]
        modules += [nn.ReLU()]
        return nn.Sequential(*modules)

    def _initialise_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layer1(x)
        x = self._layer2(x)
        x = self._classifier(x)
        return x


class MockUpAnnotator(Annotator):
    """A class demonstrating what to define when creating ``Annotator``s."""

    def __init__(self):
        name = 'MUAnnotator'
        super(MockUpAnnotator, self).__init__(name)

    def apply(self, g: fx.GraphModule) -> fx.GraphModule:
        print(f"Applying {self.__class__.__name__} ...")
        return g


class MockUpRewriter(Rewriter):
    """A class demonstrating what to define when creating ``Rewriter``s."""

    def __init__(self):
        name = 'MURewriter'
        super(MockUpRewriter, self).__init__(name)

    def find(self, g: fx.GraphModule) -> List[ApplicationPoint]:
        mockup_application_point_core = {next(iter(g.graph.nodes)): next(iter(g.graph.nodes))}
        mockup_application_point = ApplicationPoint(rewriter=self, graph=g, apcore=mockup_application_point_core)
        return [mockup_application_point]

    def _check_aps(self, g: fx.GraphModule, aps: List[ApplicationPoint]) -> None:
        if not all(map(lambda ap: ap.rewriter is self, aps)):
            raise ValueError(quantlib_err_header(obj_name=self.__class__.__name__) + "can not be applied to application points computed by other Rewritings.")

    def _apply(self, g: fx.GraphModule, ap: ApplicationPoint) -> fx.GraphModule:
        print(f"Applying {self.__class__.__name__} ...")
        return g


class MockUpComposedEditor(ComposedEditor):
    """A class demonstrating what to define when creating ``ComposedEditor``s."""

    def __init__(self):

        annotator = MockUpAnnotator()
        rewriter  = MockUpRewriter()

        super(MockUpComposedEditor, self).__init__(editors=[annotator, rewriter])


class EditorsTest(unittest.TestCase):
    # TODO: remove duplicated network creation from tests (e.g., by copying `fx.GraphModule`s) before processing them with `Editor`s

    def test_annotator(self):
        """Exemplify the annotation API usage."""

        # sub-case 1: explicit `apply` interface
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qeg.qmodule_symbolic_trace(root=net)
        # create the `Annotator`
        a = MockUpAnnotator()
        # annotate
        gmnet_annotated = a.apply(gmnet)
        self.assertTrue(isinstance(gmnet_annotated, fx.GraphModule))

        # sub-case 2: `__call__` dunder (i.e., double-underscore) method
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qeg.qmodule_symbolic_trace(root=net)
        # create the `Annotator`
        a = MockUpAnnotator()
        # annotate
        gmnet_annotated = a(gmnet)
        self.assertTrue(isinstance(gmnet_annotated, fx.GraphModule))

    def test_rewriter(self):
        """Exemplify the rewriting API usage."""

        # sub-case 1.1: explicit `apply` interface; no application point
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qeg.qmodule_symbolic_trace(root=net)
        # create the `Rewriter`
        r = MockUpRewriter()
        # rewrite (all)
        gmnet_rewritten = r.apply(gmnet)
        self.assertTrue(isinstance(gmnet_rewritten, fx.GraphModule))

        # sub-case 1.2: explicit `apply` interface; single application point
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qeg.qmodule_symbolic_trace(root=net)
        # create the `Rewriter`
        r = MockUpRewriter()
        # find application points
        aps = r.find(gmnet)
        # select an application point and rewrite it
        ap = aps.pop()
        gmnet_rewritten = r.apply(gmnet, ap)
        self.assertTrue(isinstance(gmnet_rewritten, fx.GraphModule))

        # sub-case 1.3: explicit `apply` interface; multiple application points
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qeg.qmodule_symbolic_trace(root=net)
        # create the `Rewriter`
        r = MockUpRewriter()
        # find application points
        aps = r.find(gmnet)
        # select a list of application points and rewrite them
        aps = aps  # in this case, we simply select all the found application points
        gmnet_rewritten = r.apply(gmnet, aps)
        self.assertTrue(isinstance(gmnet_rewritten, fx.GraphModule))

        # sub-case 2: `__call__` dunder (i.e., double-underscore) method
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qeg.qmodule_symbolic_trace(root=net)
        # create the `Rewriter`
        r = MockUpRewriter()
        # rewrite (all)
        gmnet_annotated = r(gmnet)
        self.assertTrue(isinstance(gmnet_annotated, fx.GraphModule))

        # sub-case 3: error when `apply`ing a `Rewriter` to the application points computed by a different `Rewriter`
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qeg.qmodule_symbolic_trace(root=net)
        # create the `Rewriter`s
        r1 = MockUpRewriter()
        r2 = MockUpRewriter()
        # find application points
        aps = r1.find(gmnet)
        # select an application point and rewrite it
        ap = next(iter(aps))
        self.assertRaises(ValueError, lambda: r2.apply(gmnet, ap))

    def test_composed_editor(self):
        """Exemplify the usage of ``ComposedEditor``s."""

        # sub-case 1: explicit `apply` interface
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qeg.qmodule_symbolic_trace(root=net)
        # create the `ComposedEditor`
        c = MockUpComposedEditor()
        # edit
        gmnet_new = c.apply(gmnet)
        self.assertTrue(isinstance(gmnet_new, fx.GraphModule))

        # sub-case 2: `__call__` dunder (i.e., double-underscore) method
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qeg.qmodule_symbolic_trace(root=net)
        # create the `ComposedEditor`
        c = MockUpComposedEditor()
        # edit
        gmnet_new = c(gmnet)
        self.assertTrue(isinstance(gmnet_new, fx.GraphModule))
