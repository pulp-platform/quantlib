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
