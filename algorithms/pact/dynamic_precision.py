# 
# pact_controllers.py
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

from functools import partial

import numpy as np
from numpy.random import default_rng


__all__ = ["select_levels_const",
           "select_levels_static",
           "select_levels_uniform"]
_rng = default_rng()

def select_levels_const(n_levels : int):
     return lambda l, _ : n_levels

def select_levels_static(p : list):
 return lambda l, _ : _rng.choice(l, p=p)

def select_levels_uniform():
    return select_levels_static(p=None)
