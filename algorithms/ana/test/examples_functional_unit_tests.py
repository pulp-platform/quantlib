# 
# examples_functional_unit_tests.py
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

from .create_modules        import create_quantizer_spec, ANAModuleFactory
from .create_test_units     import FunctionalEquivalenceUnitGenerator
# >>> from .create_tensors        import BatchSize, InputSize
# >>> from .create_test_units     import TestModule
# >>> from .functional_unit_tests import numerical_equivalence, visual_equivalence


def get_ana_feug() -> FunctionalEquivalenceUnitGenerator:
    """Create a generator of functional equivalence test units for ternary ANA
    modules.
    """
    qs       = create_quantizer_spec(2, True, True, 1.0)  # the quantizer is the only hyper-parameter that is fixed in the beginning
    amf      = ANAModuleFactory(qs)
    ana_feug = FunctionalEquivalenceUnitGenerator(amf)
    return ana_feug

# 3 batch sizes (BatchSize): SINGLE, SMALL, LARGE
# 3 input sizes (InputSize): SMALL, NORMAL, LARGE
# 4 test modules (TestModule): ACTIVATIONLINEAR, ACTIVATIONCONV2D, LINEAR, CONV2D
# 4 noise types (str): 'uniform', 'triangular', 'normal', 'logistic'
# arbitrary mi (float)
# arbitraty sigma (float)
# 2 forward computation strategies (str): 'expectation', 'mode'
# -------------------------------------------------------------------------------
# 3 * 3 * 4 * 4 * 2 = 288 tests
# -------------------------------------------------------------------------------
# >>> (x_gen_cpu, module_cpu, grad_gen_cpu), (x_gen_gpu, module_gpu, grad_gen_gpu) = feug.get_test_unit(BatchSize.SMALL, InputSize.NORMAL, TestModule.ACTIVATIONLINEAR, 'uniform', -0.5, 1.0, 'expectation')
# >>> numerical_equivalence(x_gen_cpu, module_cpu, grad_gen_cpu, x_gen_gpu, module_gpu, grad_gen_gpu)

# 3 batch sizes (BatchSize): SINGLE, SMALL, LARGE
# 3 input sizes (InputSize): SMALL, NORMAL, LARGE
# 2 test modules (TestModule): ACTIVATIONLINEAR, ACTIVATIONCONV2D
# 4 noise types (str): 'uniform', 'triangular', 'normal', 'logistic'
# arbitrary mi (float)
# arbitraty sigma (float)
# 1 forward computation strategy (str): 'random'
# ----------------------------------------------------------------------------
# 3 * 3 * 2 * 4 * 1 = 72 tests
# ----------------------------------------------------------------------------
# >>> (x_gen_cpu, module_cpu, grad_gen_cpu), (x_gen_gpu, module_gpu, grad_gen_gpu) = feug.get_test_unit(BatchSize.SMALL, InputSize.NORMAL, TestModule.ACTIVATIONCONV2D, 'uniform', -0.5, 1.0, 'random')
# >>> visual_equivalence(x_gen_cpu, module_cpu, grad_gen_cpu, x_gen_gpu, module_gpu, grad_gen_gpu)

# 3 batch sizes (BatchSize): SINGLE, SMALL, LARGE
# 3 input sizes (InputSize): SMALL, NORMAL, LARGE
# 2 test modules (TestModule): LINEAR, CONV2D
# 4 noise types (str): 'uniform', 'triangular', 'normal', 'logistic'
# arbitrary mi (float)
# arbitraty sigma (float)
# 3 forward computation strategies (str): 'expectation', 'mode', 'random'
# ----------------------------------------------------------------------------
# 3 * 3 * 2 * 4 * 3 = 216 tests
# ----------------------------------------------------------------------------
# >>> (x_gen_cpu, module_cpu, grad_gen_cpu), (x_gen_gpu, module_gpu, grad_gen_gpu) = feug.get_test_unit(BatchSize.SMALL, InputSize.NORMAL, TestModule.CONV2D, 'uniform', -0.5, 1.0, 'mode')
# >>> visual_equivalence(x_gen_cpu, module_cpu, grad_gen_cpu, x_gen_gpu, module_gpu, grad_gen_gpu)

