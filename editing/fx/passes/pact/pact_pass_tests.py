import unittest
from unittest import TestCase

import torch
from torch import nn, fx

from torchvision.models import MobileNetV2

from quantlib.editing.fx.passes.pact.canonicalize import *
from quantlib.editing.fx.passes.pact.pact_util import *

import ipdb


class NastyAdder(nn.Module):
    def __init__(self):
        super(NastyAdder, self).__init__()
        self.conv1 = nn.Conv1d(1,1,4)
        self.conv2 = nn.Conv1d(1,1,4)
        self.conv3 = nn.Conv1d(1,1,4)
        self.conv4 = nn.Conv1d(1,1,4)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        y1 = x1 + x2
        y2 = y1 + x3
        y3 = y1 + x4
        y3 = y3 + 1.3
        y4 = y1 + y2
        return y4 + y3


class TestCanonicalize(TestCase):
    # due to float magic, there will be small mismatches between the original
    # and the modified graph
    eps = 1e-6

    def test_add_tree(self):
        adder_tree = NastyAdder()
        dummy_in = torch.rand(16, 1, 32)
        out_golden = adder_tree(dummy_in)
        tree_traced = fx.symbolic_trace(adder_tree)
        # print("Initial module:")
        # print(tree_traced)
        # print("Initial graph:")
        # print(tree_traced.graph)
        add_pass = AddTreeReplacementPass()
        tree_passed = add_pass.apply(tree_traced)

        out_test = tree_passed(dummy_in)
        # print("Module after pass:")
        # print(tree_passed)
        # print("Graph after pass:")
        # print(tree_passed.graph)
        # print("Difference between passed output and golden output:")
        # print(out_test - out_golden)
        # tree_retraced = PACT_symbolic_trace(tree_passed)
        # print("Module after retrace:")
        # print(tree_retraced)
        # print("Graph after retrace:")
        # print(tree_retraced.graph)
        # out_retraced = tree_retraced(dummy_in)
        # print("Difference between retraced output and golden output:")
        # print(out_retraced - out_golden)
        # tree_retraced2 = PACT_symbolic_trace_inclusive(tree_passed)
        # print("Module after inclusive retrace:")
        # print(tree_retraced2)
        # print("Graph after inclusive retrace:")
        # print(tree_retraced2.graph)
        # out_retraced2 = tree_retraced2(dummy_in)
        # print("Difference between inclusive-retraced output and golden output:")
        # print(out_retraced2-out_golden)
        self.assertTrue(bool(torch.all(torch.abs(out_test-out_golden) < self.eps)))

    def test_mnv2(self):
        model = MobileNetV2()
        dummy_in = torch.rand(1, 3, 224, 224)
        model_traced = fx.symbolic_trace(model)
        add_pass = AddTreeReplacementPass()
        model_passed = add_pass.apply(model_traced)

        if torch.cuda.is_available():
            model = model.to('cuda')
            model_passed = model_passed.to('cuda')
            dummy_in = dummy_in.to('cuda')

        model.eval()
        model_passed.eval()

        out_golden = model(dummy_in)
        out_test = model_passed(dummy_in)
        self.assertTrue(bool(torch.all(torch.abs(out_test-out_golden) < self.eps)))

if __name__ == "__main__":
    unittest.main()
