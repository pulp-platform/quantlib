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

from typing import Union, Optional, Tuple, List, Literal
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

from .pact_util import PACT_OPS, PACT_OPS_INCLUSIVE, PACTTracer, PACT_symbolic_trace, PACT_symbolic_trace_inclusive

class ApproximateSoftmaxPass(SequentialPass):
    def __init__(self, mode: Literal["I-BERT", "ITA", 'ITA-Partial'] = "I-BERT", **kwargs):
        passes = []
        pattern = nn.Sequential(nn.Softmax())

        if mode=='I-BERT':
            replacement_class = PACTSoftmax()
        elif mode=='ITA':
            replacement_class = PACTITAMax()
        elif mode=='ITA-Partial':
            replacement_class = PACTITAPartialMax()
        else:
            assert False, f"[ApproximateSoftmaxPass] Invalid mode {mode} specified!"
    
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, lambda x,y: replacement_class, f'_APPROXIMATE_SOFTMAX_PASS'))

        super().__init__(*passes, name_prefix='_APPROXIMATE_SOFTMAX_PASS')

class ApproximateGELUPass(SequentialPass):
    def __init__(self, **kwargs):
        passes = []
        pattern = nn.Sequential(nn.GELU())
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, lambda x,y: PACTGELU(), f'_APPROXIMATE_GELU_PASS'))
        super().__init__(*passes, name_prefix='_APPROXIMATE_GELU_PASS')

def layernorm_replacement_fun(gm : fx.GraphModule, match : Match, *args, **kwargs):
    modules = gm_modules(gm)
    matched_nodes = [m for k, m in match.nodes_map.items() if k.op == 'call_module']
    layernorm_node = matched_nodes[0]
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    layernorm = matched_modules[0]
    assert isinstance(layernorm, nn.LayerNorm), f"layernorm_replacement_fun got bad match - expected LayerNorm, got {type(layernorm)}"

    weight = layernorm._parameters['weight'].clone() if layernorm._parameters['weight'] is not None  else torch.Tensor((1.,))
    bias = layernorm._parameters['bias'].clone() if layernorm._parameters['bias'] is not None else torch.Tensor((0.,))

    new_layernorm = PACTLayerNorm(layernorm.normalized_shape, weight, bias, layernorm.eps, *args, **kwargs)

    return new_layernorm

class CanonicalizeLayerNormPass(SequentialPass):
    def __init__(self, *args, **kwargs):
        passes = []
        pattern = nn.Sequential(nn.LayerNorm(1))
        passes.append(ReplaceSequentialPatternPass(pattern, PACT_symbolic_trace, partial(layernorm_replacement_fun, *args, **kwargs), f'_CANONICALIZE_LAYERNORM_PASS'))
        super().__init__(*passes, name_prefix='_CANONICALIZE_LAYERNORM_PASS')


def embedding_replacement_fun(gm : fx.GraphModule, match : Match, n_levels: int = 256):
    modules = gm_modules(gm)

    def fetch_attr(target : str):
        target_atoms = target.split('.')
        attr_itr = gm
        for i, atom in enumerate(target_atoms):
            if not hasattr(attr_itr, atom):
                raise RuntimeError(f"Node referenced nonexistant target {'.'.join(target_atoms[:i])}")
            attr_itr = getattr(attr_itr, atom)
        return attr_itr

    matched_embedding = [m for k,m in match.nodes_map.items() if k.op =='getattr' or k.op == 'get_attr']
    bias = fetch_attr(matched_embedding[0].target)

    new_embedding = PACTEmbedding(n_levels, bias)

    return new_embedding

class ProtoPACTEmbedding(torch.nn.Module):

    def __init__(self, weights : torch.Tensor = torch.Tensor((1.,))):
        super().__init__()
        self.weights = nn.Parameter(weights)
        self.adder = PACTIntegerAdd(n_levels=256, num_args=2, act_kind='identity', init_clip='max', learn_clip=True)

    def forward(self, x):
        out = self.adder(x, self.weights)
        return out

# This can be made much more general -- Current workaround
class CanonicalizeEmbeddingsPass(SequentialPass):
    def __init__(self, n_levels:int = 256, **kwargs):
        passes = []
        # Use IntegerEmbedding to get matches back since otherwise it doesn't get traced right
        pattern = nn.Sequential(ProtoPACTEmbedding(torch.Tensor((1.,))))
        passes.append(ReplaceSingleInputPatternPass(pattern, PACT_symbolic_trace_inclusive, partial(embedding_replacement_fun, n_levels=n_levels), f'_CANONICALIZE_EMBEDDING_PASS'))
        super().__init__(*passes, name_prefix='_CANONICALIZE_EMBEDDING_PASS')
