# 
# __init__.py
# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
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

from .create_tensors         import BatchSize, InputSize
from .create_test_units      import TestModule
# from .create_modules         import FloatingPointModuleFactory, ANAModuleFactory
# from .create_modules         import create_quantizer_spec
# from .create_test_units      import FunctionalEquivalenceUnitGenerator
# from .create_test_units      import ProfilingUnitGenerator
from .functional_unit_tests  import numerical_equivalence, visual_equivalence
from .performance_unit_tests import profile

from .examples_functional_unit_tests  import get_ana_feug
from .examples_performance_unit_tests import get_floating_pug, get_ana_pug

