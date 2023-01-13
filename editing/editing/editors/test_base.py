# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 

import unittest
import random
import torch
import torch.nn as nn
import torch.fx as fx
from typing import List

from .base.editor import Editor
from .base import Annotator
from .base import ApplicationPoint, Finder, Applier, Rewriter
from .base import ComposedEditor
import quantlib.editing.graphs as qg


# -- 1. ANNOTATOR -- #

class MockUpAnnotator(Annotator):
    """A class demonstrating how to define an ``Annotator``."""

    def __init__(self):
        name: str = 'MUAnnotator'
        symbolic_trace_fn: qg.fx.SymbolicTraceFnType = qg.fx.quantlib_symbolic_trace
        super(MockUpAnnotator, self).__init__(name=name,
                                              symbolic_trace_fn=symbolic_trace_fn)

    def apply(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:
        return g


# -- 2. REWRITER -- #

class MockUpApplicationPoint(ApplicationPoint):
    """A class demonstrating how to define an ``ApplicationPoint``."""
    pass


class MockUpFinder(Finder):
    """A class demonstrating how to define a ``Finder``."""
    
    def find(self, g: fx.GraphModule) -> List[MockUpApplicationPoint]:
        # create fake application points
        _MIN_APPLICATION_POINTS: int = 1
        _MAX_APPLICATION_POINTS: int = 10
        return [MockUpApplicationPoint() for _ in range(0, random.randint(_MIN_APPLICATION_POINTS, _MAX_APPLICATION_POINTS))]
    
    def check_aps_commutativity(self, aps: List[MockUpApplicationPoint]) -> bool:
        return True


class MockUpApplier(Applier):
    """A class demonstrating how to define an ``Applier``."""
    
    def _apply(self, g: fx.GraphModule, ap: MockUpApplicationPoint, id_: str) -> fx.GraphModule:
        return g


class MockUpRewriter(Rewriter):
    """A class demonstrating how to define a ``Rewriter``."""

    def __init__(self):
        name = 'MURewriter'
        symbolic_trace_fn: qg.fx.SymbolicTraceFnType = qg.fx.quantlib_symbolic_trace
        finder = MockUpFinder()
        applier = MockUpApplier()
        super(MockUpRewriter, self).__init__(name=name,
                                             symbolic_trace_fn=symbolic_trace_fn,
                                             finder=finder,
                                             applier=applier)


# -- 3. COMPOSED EDITOR -- #

class MockUpComposedEditor(ComposedEditor):
    """A class demonstrating how to define a ``ComposedEditor``."""

    def __init__(self):

        annotator: Editor = MockUpAnnotator()
        rewriter: Editor = MockUpRewriter()

        super(MockUpComposedEditor, self).__init__(children_editors=[annotator, rewriter])


# -- TESTS -- #

class MLP(nn.Module):
    """A simple multi-layer perceptron to test editing flows."""

    def __init__(self):

        super(MLP, self).__init__()

        self._layer1 = MLP._make_layer(64, 64)
        self._layer2 = MLP._make_layer(64, 64)
        self._classifier = nn.Linear(in_features=64, out_features=10, bias=False)

        self._initialise_parameters()

    @staticmethod
    def _make_layer(in_features: int, out_features: int) -> nn.Sequential:
        modules = []
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


class EditorsTest(unittest.TestCase):

    def test_annotator(self):
        """Exemplify the annotation API usage."""

        # sub-case 1: explicit `apply` interface
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qg.fx.quantlib_symbolic_trace(root=net)
        # create the `Annotator`
        a = MockUpAnnotator()
        # annotate
        gmnet_annotated = a.apply(gmnet)
        self.assertTrue(isinstance(gmnet_annotated, fx.GraphModule))

        # sub-case 2: `__call__` dunder (i.e., double-underscore) method
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qg.fx.quantlib_symbolic_trace(root=net)
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
        gmnet = qg.fx.quantlib_symbolic_trace(root=net)
        # create the `Rewriter`
        r = MockUpRewriter()
        # rewrite (all)
        gmnet_rewritten = r.apply(gmnet)
        self.assertTrue(isinstance(gmnet_rewritten, fx.GraphModule))

        # sub-case 1.2: explicit `apply` interface; single application point
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qg.fx.quantlib_symbolic_trace(root=net)
        # create the `Rewriter`
        r = MockUpRewriter()
        # find application points
        aps = r.find(gmnet)
        # select an application point and rewrite it
        ap = random.choice(aps)
        gmnet_rewritten = r.apply(gmnet, ap)
        self.assertTrue(isinstance(gmnet_rewritten, fx.GraphModule))

        # sub-case 1.3: explicit `apply` interface; multiple application points
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qg.fx.quantlib_symbolic_trace(root=net)
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
        gmnet = qg.fx.quantlib_symbolic_trace(root=net)
        # create the `Rewriter`
        r = MockUpRewriter()
        # rewrite (all)
        gmnet_annotated = r(gmnet)
        self.assertTrue(isinstance(gmnet_annotated, fx.GraphModule))

        # sub-case 3: error when `apply`ing a `Rewriter` to the application points computed by a different `Rewriter`
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qg.fx.quantlib_symbolic_trace(root=net)
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
        gmnet = qg.fx.quantlib_symbolic_trace(root=net)
        # create the `ComposedEditor`
        c = MockUpComposedEditor()
        # edit
        gmnet_new = c.apply(gmnet)
        self.assertTrue(isinstance(gmnet_new, fx.GraphModule))

        # sub-case 2: `__call__` dunder (i.e., double-underscore) method
        # create the target `fx.GraphModule`
        net = MLP()
        gmnet = qg.fx.quantlib_symbolic_trace(root=net)
        # create the `ComposedEditor`
        c = MockUpComposedEditor()
        # edit
        gmnet_new = c(gmnet)
        self.assertTrue(isinstance(gmnet_new, fx.GraphModule))
