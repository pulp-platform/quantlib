# 
# util.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
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

from torch import nn
import torch
from .pact_functions import PACTQuantize

def assert_param_valid(module : nn.Module, value, param_name : str, valid_values : list):
    error_str = f"[{module.__class__.__name__}]  Invalid argument {param_name}: Got {value}, expected {valid_values[0] if len(valid_values)==1 else ', '.join(valid_values[:-1]) + ' or ' + str(valid_values[-1])}"
    assert value in valid_values, error_str

def almost_symm_quant(max_val, n_levels):
    # for binary quantization, we want -eps/2 and +eps/2!
    if n_levels == 2:
        return -max_val, max_val

    if n_levels % 2 == 0:
        eps = 2*max_val/n_levels
    else:
        eps = 2*max_val/(n_levels-1)
    min_val = -max_val
    max_val = min_val + (n_levels-1)*eps
    return min_val, max_val


# implemented like in MQBench: https://github.com/ModelTC/MQBench/blob/main/mqbench/observer.py
def mse_bounds(x, n_levels : int, signed : bool, channelwise : bool, x_is_weights : bool, n_iters : int = 80, symm : bool = True, rounding : bool = True):
    if channelwise and not x_is_weights:
        x = x.permute(1, 0, 2, 3)
    reduce_dims = tuple(range(len(x.shape)))
    if channelwise:
        reduce_dims = reduce_dims[1:]

    act_max = x.amax(dim=reduce_dims, keepdim=True)
    if signed:
        act_min = x.amin(dim=reduce_dims, keepdim=True)
    else:
        act_min = torch.zeros_like(act_max)

    best_min, best_max = act_min, act_max
    best_dist = 1000000

    for i in range(n_iters):
        cur_min = act_min * (1 - i * 0.01)
        cur_max = act_max * (1 - i * 0.01)
        if symm and signed:
            abs_max = torch.maximum(-cur_min, cur_max)
            cur_min, cur_max = almost_symm_quant(abs_max, n_levels)
        with torch.no_grad():
            eps = (cur_max - cur_min)/(n_levels-1)
            x_quant = PACTQuantize(x, eps, cur_min, cur_max, floor=(not rounding), clip_gradient=False, noisy=False)
            cur_dist = (x-x_quant).pow(2).mean()
            if cur_dist < best_dist:
                best_dist = cur_dist
                best_min = cur_min
                best_max = cur_max

    return best_min.squeeze(), best_max.squeeze()
