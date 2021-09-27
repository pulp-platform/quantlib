# ----------------------------------------------------------------------
#
# File: approximate.py
#
# Last edited: 23.09.2021        
# 
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Author: Moritz Scherer, ETH Zurich
#
# ----------------------------------------------------------------------
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the License); you may
# not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an AS IS BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Union, Optional, Tuple, List
from dataclasses import dataclass
from functools import partial

import numpy as np

import torch
from torch import fx, nn
from torch.fx.subgraph_rewriter import Match

from quantlib.algorithms.pact.pact_ops import *

from .. import FxPass, ReplaceSequentialPatternPass, ModifySequentialPatternPass, SequentialPass, ShapePropPass
from .. import AnnotateEpsPass, extract_eps
from .. import MergeConvBNPass, RetracePass
from ...util import gm_modules, module_of_node
from ...util.tracing import LeafTracer, custom_symbolic_trace

from .pact_util import PACT_OPS, PACT_OPS_INCLUSIVE, PACTTracer, PACT_symbolic_trace
    
class ApproximateSoftmaxPass(SequentialPass):
    def __init__(self, n_levels, **kwargs):
        passes = []
        pattern = nn.Sequential(nn.Softmax())
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, lambda x,y: PACTSoftmax(n_levels), f'_APPROXIMATE_SOFTMAX_PASS'))
        super().__init__(*passes, name_prefix='_APPROXIMATE_SOFTMAX_PASS')
        
class ApproximateGELUPass(SequentialPass):
    def __init__(self, n_levels, **kwargs):
        passes = []
        pattern = nn.Sequential(nn.GELU())
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, lambda x,y: PACTGELU(n_levels), f'_APPROXIMATE_GELU_PASS'))
        super().__init__(*passes, name_prefix='_APPROXIMATE_GELU_PASS')

class CanonicalizeLayerNormPass(SequentialPass):
    def __init__(self, n_levels, **kwargs):
        passes = []
        pattern = nn.Sequential(nn.LayerNorm(1))
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, lambda x,y: PACTLayerNorm(n_levels), f'_CANONICALIZE_LAYERNORM_PASS'))
        super().__init__(*passes, name_prefix='_CANONICALIZE_LAYERNORM_PASS')
