from .create_tensors        import BatchSize, InputSize
from .create_test_units     import TestModule
from .create_modules        import create_quantizer_spec
from .create_modules        import ANAModuleFactory
from .create_test_units     import FunctionalEquivalenceUnitGenerator
from .functional_unit_tests import numerical_equivalence, visual_equivalence


qs = create_quantizer_spec(2, True, True, 1.0)  # the quantizer is the only hyper-parameter that is fixed in the beginning
amf = ANAModuleFactory(qs)
feug = FunctionalEquivalenceUnitGenerator(amf)

# 3 batch sizes (BatchSize): SINGLE, SMALL, LARGE
# 3 input sizes (InputSize): SMALL, NORMAL, LARGE
# 4 test modules (TestModule): ACTIVATIONLINEAR, ACTIVATIONCONV2D, LINEAR, CONV2D
# 4 noise types (str): 'uniform', 'triangular', 'normal', 'logistic'
# arbitrary mi (float)
# arbitraty sigma (float)
# 2 forward computation strategies (int): 0 (expecation), 1 (mode)
# -------------------------------------------------------------------------------
# 4 * 2 * 3 * 3 * 4 = 288 tests
(x_gen_cpu, module_cpu, grad_gen_cpu), (x_gen_gpu, module_gpu, grad_gen_gpu) = feug.get_test_unit(BatchSize.SMALL, InputSize.NORMAL, TestModule.ACTIVATIONLINEAR, 'uniform', -0.5, 1.0, 0)
numerical_equivalence(x_gen_cpu, module_cpu, grad_gen_cpu, x_gen_gpu, module_gpu, grad_gen_gpu)

# 3 batch sizes (BatchSize): SINGLE, SMALL, LARGE
# 3 input sizes (InputSize): SMALL, NORMAL, LARGE
# 2 test modules (TestModule): ACTIVATIONLINEAR, ACTIVATIONCONV2D
# 4 noise types (str): 'uniform', 'triangular', 'normal', 'logistic'
# arbitrary mi (float)
# arbitraty sigma (float)
# 3 forward computation strategies (int): 0 (expecation), 1 (mode), 2 (random)
# ----------------------------------------------------------------------------
# 4 * 3 * 3 * 3 * 2 = 216 tests
(x_gen_cpu, module_cpu, grad_gen_cpu), (x_gen_gpu, module_gpu, grad_gen_gpu) = feug.get_test_unit(BatchSize.SMALL, InputSize.NORMAL, TestModule.ACTIVATIONCONV2D, 'uniform', -0.5, 1.0, 2)
visual_equivalence(x_gen_cpu, module_cpu, grad_gen_cpu, x_gen_gpu, module_gpu, grad_gen_gpu)

# 3 batch sizes (BatchSize): SINGLE, SMALL, LARGE
# 3 input sizes (InputSize): SMALL, NORMAL, LARGE
# 2 test modules (TestModule): LINEAR, CONV2D
# 4 noise types (str): 'uniform', 'triangular', 'normal', 'logistic'
# arbitrary mi (float)
# arbitraty sigma (float)
# 3 forward computation strategies (int): 0 (expecation), 1 (mode), 2 (random)
# ----------------------------------------------------------------------------
# 4 * 3 * 3 * 3 * 2 = 216 tests
(x_gen_cpu, module_cpu, grad_gen_cpu), (x_gen_gpu, module_gpu, grad_gen_gpu) = feug.get_test_unit(BatchSize.SMALL, InputSize.NORMAL, TestModule.CONV2D, 'uniform', -0.5, 1.0, 2)
visual_equivalence(x_gen_cpu, module_cpu, grad_gen_cpu, x_gen_gpu, module_gpu, grad_gen_gpu)
