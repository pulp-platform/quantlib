import torch
import numpy as np
from ..controller import Controller
from .pact_ops import PACT_UnsignedAct, PACT_AsymmetricAct, PACT_Conv1d, PACT_Conv2d
from typing import Union

__all__ = ['PACT_ActController', 'PACT_LinearController']


# Coefficients generated with this code:
# https://colab.research.google.com/drive/1IH8TwCfkvcMkHx4tHldMNgsvBIgCpTeM?usp=sharing
# first entry is c1, second is c2 with alpha_w^* = c1 * sqrt(Ew2) - c2 * Ew1
__sawb_asymm_lut = {
    2: [1.708, 0.403],
    3: [4.712, 3.621],
    4: [8.536, 7.931],
    5: [12.251, 12.201],
    6: [14.858, 15.296],
    7: [20.552, 22.297],
    8: [28.221, 31.858]
}


def almost_symm_quant(max_val, n_levels):
    if n_levels % 2 == 0:
        eps = 2*max_val/n_levels
    else:
        eps = 2*max_val/(n_levels-1)
    min_val = -max_val
    max_val = min_val + (n_levels-1)*eps
    return min_val, max_val

class PACT_ActController(Controller):
    """
    Controller for PACT activation classes:
      - PACT_UnsignedAct
      - PACT_AsymmetricAct
    The controller takes a schedule telling it to enable/disable quantization
    at given epochs. Starting quantization entails setting the `quantize` flags
    to `True` and setting the initial clipping values based on the `init_clip` value.
    """
    def __init__(self, modules : list, schedule : dict, verbose : bool = False):
        super(PACT_ActController, self).__init__()
        assert all(isinstance(m, (PACT_AsymmetricAct, PACT_UnsignedAct)) for m in modules), "Non-activation modules passed to PACT_ActController!"
        self.modules = modules
        self.schedule = {int(k):v.lower() for k,v in schedule.items()}
        self.verbose = verbose

    def step_pre_training(self, epoch : int, *args, **kwargs):
        """
        Executed before every training epoch. If the current epoch is in the schedule, perform the indicated action:
        - 'start': start quantization - set the clipping bound(s) according to the method set as `init_clip` and set
          `quantize` to True
        - 'freeze': set `requires_grad` to False for all clipping parameters so they are not updated anymore
        - 'thaw':   set `requires_grad` to True for all clipping parameters
        - 'stop':   set `quantize` to False and set `requires_grad` to False for all clipping parameters
        :param epoch: The current epoch
        """
        def set_learn_clip(val : bool):
            for m in self.modules:
                for p in m.clipping_params.values():
                    # TODO check if m.symm...
                    p.requires_grad = val

        if epoch in self.schedule.keys():
            cmd = self.schedule[epoch]
            self.log("Epoch {} - running command {}".format(epoch, cmd))
            if cmd == 'verbose_on':
                self.verbose = True
                self.log("Verbose Mode Enabled!")
            elif cmd == 'verbose_off':
                self.verbose = False
            elif cmd == 'start':
                for m in self.modules:
                    m.reset_clip_bounds(m.init_clip)
                    m.started = True
                self.log("Started Quantization!")
            elif cmd == 'freeze':
                set_learn_clip(False)
                self.log("Froze Clipping Parameters!")
            elif cmd == 'thaw':
                # 'thaw' means enable learning for clipping parameters
                set_learn_clip(True)
                self.log("Unfroze Clipping Parameters!")
            else:
                assert cmd == 'stop', "Invalid PACT_ActController command at epoch {}: {}".format(epoch, cmd)
                for m in self.modules:
                    m.started = False
                self.log("Stopped Quantization!")

    def reset_clip_bounds(self, m : Union[PACT_UnsignedAct, PACT_AsymmetricAct], method : str = None):
        if not method:
            method = m.init_clip
        if method == 'max':
            max_val = m.max.data
            try:
                if m.symm:
                    max_val = torch.maximum(max_val, -m.min)
                    min_val, max_val = almost_symm_quant(max_val, m.n_levels)
                else:
                    min_val = m.min.data
            except AttributeError:
                # we don't need 'min_val' if m does not have the 'symm' attribute because in that case
                # it's not an asymmetric activation
                pass
        else: # method == 'std'
            max_val = m.running_mean.data + m.nb_std * torch.sqrt(m.running_var.data)
            try:
                if m.symm:
                    min_val, max_val = almost_symm_quant(max_val, m.n_levels)
                else:
                    min_val = m.running_mean.data - m.nb_std * torch.sqrt(m.running_var.data)
            except AttributeError:
                pass

        for k, b in m.clipping_params.items():
            if k == 'high':
                b.data = max_val
            elif k == 'low':
                b.data = min_val
            else:
                assert False, "Unexpected clipping parameter dictionary key in module of type {}: {}".format(type(m), k)

    def log(self, msg : str):
        if self.verbose:
            print("[PACT_ActController]   ", msg)



class PACT_LinearController(Controller):
    """
    Controller for PACT Linear classes:
      - PACT_Conv2d
      - PACT_Conv1d
      - PACT_Linear
    """

    def __init__(self, modules : list, schedule : dict, verbose : bool = False):
        super(PACT_LinearController, self).__init__()
        self.modules = modules
        self.schedule = {int(k):v.lower() for k,v in schedule.items()}
        self.verbose = verbose

    def step_pre_batch(self, *args, **kwargs):
        with torch.no_grad():
            for m in self.modules:
                if (not m.frozen) and m.started:
                    if not m.learn_clip:
                        self.reset_clip_bounds(m)
                    # if 'learn_clip' is True and 'symm_wts' is also True, we learn the lower clip bound and set the upper
                    # one automatically with a function equivalent to 'almost_symm_quant'. This is performed in the
                    # conv/linear module itself to ensure proper gradient propagation (???).
                    # However, we also update the upper clipping bound for layers where 'symm_wts' and 'learn_clip'
                    # here so it reflects the value that is used in forward propagation.
                    elif m.learn_clip and m.symm_wts:
                        _, max_val = almost_symm_quant(-m.clip_lo.data, m.n_levels)
                        m.clip_hi.data = max_val

    def step_pre_training(self, epoch : int, *args, **kwargs):
        if epoch in self.schedule.keys():
            cmd = self.schedule[epoch]
            self.log("Epoch {} - running command {}".format(epoch, cmd))
            if cmd == 'verbose_on':
                self.verbose = True
                self.log("Verbose Mode Enabled!")
            elif cmd == 'verbose_off':
                self.verbose = False
            elif cmd == 'start':
                for m in self.modules:
                    self.reset_clip_bounds(m)
                    m.started = True
                self.log("Started Quantization!")
            elif cmd == 'freeze':
                for m in self.modules:
                    for b in m.clipping_params.values():
                        b.requires_grad = False
                    m.frozen = True
            elif cmd == 'thaw':
                for m in self.modules:
                    for k, b in m.clipping_params.items():
                        # if symm_wts is True, the upper bound is not learned but inferred from the lower bound.
                        b.requires_grad = m.learn_clip and not (k=='high' and m.symm_wts)
                    m.frozen = False
            else:
                assert cmd == 'stop', "Invalid PACT_ConvController command at epoch {}: {}".format(epoch, cmd)
                for m in self.modules:
                    m.started = False
                self.log("Stopped Quantization!")

    def step_pre_validation(self, epoch : int, *args, **kwargs):
        # always before validation, update the clipping parameters as is done before each batch, so the changes from
        # the last batch of an epoch are reflected in the clipping params.
        self.step_pre_batch()


    # resetting clip bounds is almost identical between the different convolutions
    def reset_clip_bounds(self, m : Union[PACT_Conv2d, PACT_Conv1d]):
        method = m.init_clip
        w = m.weight.data
        if m.quantize == 'per_channel':
            reduce_dims = tuple(range(1, len(w.shape)))
        else:
            reduce_dims = tuple(range(len(w.shape)))

        if method == 'max':
            if m.symm_wts:
                max_val = torch.amax(w.abs(), dim=reduce_dims)
                # if symm_wts is true, we do "almost symmetric" quantization in the case of an even n_levels (to account for e.g. int8 range going from -128 to 127)
                min_val, max_val = almost_symm_quant(max_val, m.n_levels)
            else:
                max_val = torch.amax(w, dim=reduce_dims)
                min_val = torch.amin(w, dim=reduce_dims)
        elif method == 'std':
            #TODO this doesn't really make sense. this way, the lower bound is initialized to -mean-nb_std*std.
            # if mean is 0 (which it should approximately be...) that's fine but if not, it's not really a proper
            # mean + std initialization.
            max_val = m.running_mean + m.nb_std * torch.sqrt(m.running_var)
            # 'std' initialization is inherently symmetrical, so use the "almost_symmetrical" quantization anyway
            min_mal, max_val = almost_symm_quant(max_val, m.n_levels)
        elif method == 'sawb':
            # mean absolute weights: E[|W|] - either channel-wise or over the whole tensor depending on reduce_dims
            # when using SAWB
            e_w_abs = torch.mean(m.weight.data.abs(), dim=reduce_dims)
            e_w2 = torch.mean(m.weight.data**2, dim=reduce_dims)
            n_bits = int(np.log2(m.n_levels))
            assert int(2**n_bits) == m.n_levels, "SAWB not supported for n_levels={} - must be power of 2!".format(m.n_levels)
            coeffs = __sawb_asymm_lut[n_bits]
            alpha = coeffs[0] * torch.sqrt(e_w2) - coeffs[1] * e_w_abs
            min_val, max_val = almost_symm_quant(alpha, m.n_levels)

        m.max.data = m.expand_bounds(max_val)
        m.min.data = m.expand_bounds(min_val)

    def log(self, msg : str):
        if self.verbose:
            print("[PACT_ConvController]   ", msg)
