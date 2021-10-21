# 
# util.py
# 
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2021 ETH Zurich. All rights reserved.
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

def assert_param_valid(module : nn.Module, value, param_name : str, valid_values : list):
    error_str = f"[{module.__class__.__name__}]  Invalid argument {param_name}: Got {value}, expected {valid_values[0] if len(valid_values)==1 else ', '.join(valid_values[:-1]) + ' or ' + str(valid_values[-1])}"
    assert value in valid_values, error_str

def almost_symm_quant(max_val, n_levels):
    if n_levels % 2 == 0:
        eps = 2*max_val/n_levels
    else:
        eps = 2*max_val/(n_levels-1)
    min_val = -max_val
    max_val = min_val + (n_levels-1)*eps
    return min_val, max_val
