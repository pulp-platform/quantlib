# ----------------------------------------------------------------------
#
# File: approximate.py
#
# Last edited: 23.09.2021
#
# Copyright (C) 2021, ETH Zurich and University of Bologna.
#
# Authors: 
# - Moritz Scherer, ETH Zurich
# - Victor Jung, ETH Zurich
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

from typing import Literal, Callable, Union
from functools import partial
import torch
from torch import fx, nn
from torch.fx.subgraph_rewriter import Match

from quantlib.algorithms.pact.pact_ops import *
from .. import ReplaceSequentialPatternPass, SequentialPass
from ...util import gm_modules
from .pact_util import PACT_symbolic_trace, PACT_symbolic_trace_inclusive


def replSoftmax(gm : fx.GraphModule, match : Match, mode: str):
    if mode == "I-BERT":
        replacement_class = PACTSoftmax()
    elif mode=='ITA':
        replacement_class = PACTITAMax()
    elif mode=='ITA-Partial':
        replacement_class = PACTITAPartialMax()

    return replacement_class

class ApproximateSoftmaxPass(SequentialPass):

    modes = ["I-BERT", "ITA", 'ITA-Partial']

    def __init__(self, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace, mode: Literal["I-BERT", "ITA", 'ITA-Partial'] = "I-BERT", **kwargs):
        passes = []
        pattern = nn.Sequential(nn.Softmax())
        assert mode in self.modes, f"[ApproximateSoftmaxPass] Invalid mode {mode} specified!"
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(replSoftmax, mode=mode), f'_APPROXIMATE_SOFTMAX_PASS'))
        super().__init__(*passes, name_prefix='_APPROXIMATE_SOFTMAX_PASS')

class ApproximateGELUPass(SequentialPass):
    def __init__(self, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace, **kwargs):
        passes = []
        pattern = nn.Sequential(nn.GELU())
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, lambda x,y: PACTGELU(), f'_APPROXIMATE_GELU_PASS'))
        super().__init__(*passes, name_prefix='_APPROXIMATE_GELU_PASS')

class ApproximateSiLUPass(SequentialPass):
    def __init__(self, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace, n_levels: int = 255, **kwargs):
        passes = []
        pattern = nn.Sequential(nn.SiLU())
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, lambda x,y: PACTHardswish(eps_s=1/n_levels), f'_APPROXIMATE_SILU_PASS'))
        super().__init__(*passes, name_prefix='_APPROXIMATE_SILU_PASS')

def layernorm_replacement_fun(gm : fx.GraphModule, match : Match, *args, **kwargs):
    modules = gm_modules(gm)
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    layernorm = matched_modules[0]
    assert isinstance(layernorm, nn.LayerNorm), f"layernorm_replacement_fun got bad match - expected LayerNorm, got {type(layernorm)}"

    weight = layernorm._parameters['weight'].clone() if layernorm._parameters['weight'] is not None  else torch.Tensor((1.,))
    bias = layernorm._parameters['bias'].clone() if layernorm._parameters['bias'] is not None else torch.Tensor((0.,))

    new_layernorm = PACTLayerNorm(layernorm.normalized_shape, weight, bias, layernorm.eps, *args, **kwargs)

    return new_layernorm

class CanonicalizeLayerNormPass(SequentialPass):
    def __init__(self, symbolic_trace: Callable[[Union[nn.Module, fx.GraphModule]], fx.GraphModule] = PACT_symbolic_trace, *args, **kwargs):
        passes = []
        pattern = nn.Sequential(nn.LayerNorm(1))
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(layernorm_replacement_fun, *args, **kwargs), f'_CANONICALIZE_LAYERNORM_PASS'))
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

def rmsnorm_replacement_fun(custom_module, gm : fx.GraphModule, match : Match, *args, **kwargs):
    modules = gm_modules(gm)
    matched_modules = [modules[m.target] for k, m in match.nodes_map.items() if k.op == 'call_module'][::-1]
    rmsnorm = matched_modules[0]

    assert isinstance(rmsnorm, type(custom_module)), f"rmsnorm_replacement_fun got bad match - expected LayerNorm, got {type(rmsnorm)}"

    weight = rmsnorm._parameters['weight'].clone() if rmsnorm._parameters['weight'] is not None  else torch.Tensor((1.,))

    new_rmsnorm = PACTRMSNorm(rmsnorm.weight.shape, weight, rmsnorm.variance_epsilon)

    return new_rmsnorm

class CanonicalizeRMSNormPass(SequentialPass):
    def __init__(self, symbolic_trace: callable, custom_module, *args, **kwargs):
        passes = []
        pattern = nn.Sequential(custom_module)
        passes.append(ReplaceSequentialPatternPass(pattern, symbolic_trace, partial(rmsnorm_replacement_fun, custom_module, *args, **kwargs), f'_CANONICALIZE_RMSNORM_PASS'))
        super().__init__(*passes, name_prefix='_CANONICALIZE_RMSNORM_PASS')

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
