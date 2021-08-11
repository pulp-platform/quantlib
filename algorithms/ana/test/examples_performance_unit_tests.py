import torch

from .create_modules         import FloatingPointModuleFactory
from .create_modules         import create_quantizer_spec, ANAModuleFactory
from .create_test_units      import ProfilingUnitGenerator
# >>> from .create_tensors         import BatchSize, InputSize
# >>> from .create_test_units      import TestModule
# >>> from .performance_unit_tests import profile


def get_floating_pug(device: torch.device) -> ProfilingUnitGenerator:
    """Create a generator of profiling test units for PyTorch modules."""
    fmf          = FloatingPointModuleFactory()
    floating_pug = ProfilingUnitGenerator(fmf, device)
    return floating_pug

# 2 devices (torch.device): CPU, GPU
# 3 * 3 * 4 + 2 * 3 * 2 size/module combinations:
#     `BatchSize.SINGLE` is not compatible with `TestModule.LINEARNETWORK`
#     and `TestModule.CONV2DNETWORK`
# ------------------------------------------------------------------------
# 2 * ( 3 * 3 * 4 + 2 * 3 * 2 ) = 96 tests
# ------------------------------------------------------------------------
# >>> fx_gen, fmodule, fgrad_gen = fpug.get_test_unit(BatchSize.SINGLE, InputSize.NORMAL, TestModule.CONV2D)
# >>> profile(fx_gen, fmodule, fgrad_gen)


def get_ana_pug(device: torch.device) -> ProfilingUnitGenerator:
    """Create a generator of profiling test units for ternary ANA modules."""
    qs      = create_quantizer_spec(2, True, True, 1.0)
    amf     = ANAModuleFactory(qs)
    ana_pug = ProfilingUnitGenerator(amf, device)
    return ana_pug

# 2 devices (torch.device): CPU, GPU
# 3 * 3 * 4 + 2 * 3 * 2 size/module combinations:
#     `BatchSize.SINGLE` is not compatible with `TestModule.LINEARNETWORK`
#     and `TestModule.CONV2DNETWORK`
# 4 noise types (str): 'uniform', 'triangular', 'normal', 'logistic'
# arbitrary mi (float)
# arbitrary sigma (float)
# 3 forward computation strategies (int): 0 (expectation), 1 (mode), 2 (random)
# -----------------------------------------------------------------------------
# 2 * 4 * 3 * ( 3 * 3 * 4 + 2 * 3 * 2 ) = 1152 tests
# -----------------------------------------------------------------------------
# >>> ax_gen, amodule, agrad_gen = apug.get_test_unit(BatchSize.SMALL, InputSize.NORMAL, TestModule.LINEARNETWORK, noise_type='uniform', mi=-0.5, sigma=1.0, strategy=0)
# >>> profile(ax_gen amodule, agrad_gen
