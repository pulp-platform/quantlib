# 
# general.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich.
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
from unittest import TestCase

import torch
from torch import nn, fx

from torchvision.models import MobileNetV2

from quantlib.editing.fx.passes.pact.canonicalize import *
from quantlib.editing.fx.passes.pact.pact_util import *



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

class NastyConcat(nn.Module):
    def __init__(self, stack : bool):
        super(NastyConcat, self).__init__()
        self.conv1 = nn.Conv1d(1,1,4)
        self.conv2 = nn.Conv1d(1,1,4)
        self.conv3 = nn.Conv1d(1,1,4)
        self.conv4 = nn.Conv1d(1,1,4)
        self.stack = stack
        # stack=True doesn't actually work...
        self.cat_op = torch.stack if stack else torch.cat

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        y1 = self.cat_op([x1, x2])
        y2 = self.cat_op([y1, x3])
        y3 = self.cat_op([y1, x4])
        return self.cat_op([y1, y2, y3])


class TestCanonicalize(TestCase):
    # due to float magic, there will be small mismatches between the original
    # and the modified graph
    eps = 1e-6

    def test_add_tree(self):
        adder_tree = NastyAdder()
        dummy_in = torch.rand(16, 1, 32)
        out_golden = adder_tree(dummy_in)
        tree_traced = fx.symbolic_trace(adder_tree)
        add_pass = AddTreeReplacementPass()
        tree_passed = add_pass(tree_traced)

        out_test = tree_passed(dummy_in)
        self.assertTrue(bool(torch.all(torch.abs(out_test-out_golden) < self.eps)))

    def test_mnv2(self):
        model = MobileNetV2()
        dummy_in = torch.rand(1, 3, 224, 224)
        model_traced = fx.symbolic_trace(model)
        add_pass = AddTreeReplacementPass()
        model_passed = add_pass(model_traced)

        if torch.cuda.is_available():
            model = model.to('cuda')
            model_passed = model_passed.to('cuda')
            dummy_in = dummy_in.to('cuda')

        model.eval()
        model_passed.eval()

        out_golden = model(dummy_in)
        out_test = model_passed(dummy_in)
        self.assertTrue(bool(torch.all(torch.abs(out_test-out_golden) < self.eps)))

    def test_cat_tree(self):
        cat_tree = NastyConcat(stack=False)
        dummy_in = torch.rand(16, 1, 32)
        out_golden_cat = cat_tree(dummy_in)
        cat_pass = ConcatTreeReplacementPass()
        tree_traced_cat = fx.symbolic_trace(cat_tree)
        tree_passed_cat = cat_pass(tree_traced_cat)
        out_test_cat = tree_passed_cat(dummy_in)
        self.assertTrue(bool(torch.all(out_test_cat==out_golden_cat)))

if __name__ == "__main__":
    unittest.main()
