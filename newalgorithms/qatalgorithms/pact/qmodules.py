from enum import Enum, auto
import torch
import torch.nn as nn
from typing import Tuple, Union

from .lib import _PACTQuantiser, _PACTRedirectClipHiGrad
from quantlib.newalgorithms.qbase import get_scale, get_zero_scale
from quantlib.newutils import quantlib_err_header


class PACTLearnableClippingBounds(Enum):
    CLIP_LO_AND_CLIP_HI = auto()
    CLIP_LO             = auto()
    CLIP_HI             = auto()


class _PACTModule(nn.Module):

    def __init__(self):

        self._pact_learnable_bounds: Union[None, PACTLearnableClippingBounds] = None
        self._get_learnable_clipping_bounds()

        self.register_buffer('_is_frozen', torch.tensor(False))
        self._flag_bounds_as_learnable()
    
    def _check_clipping_bounds(self, a: torch.Tensor, b: torch.Tensor):
        if not torch.all(a < b):
            raise RuntimeError(quantlib_err_header(obj_name=self.__class__.__name__) + "some lower clipping bounds are not lower than the corresponding higher clipping bounds.")

    def _get_learnable_clipping_bounds(self):

        if self._qrange.is_quasisymmetric or self._qrange.is_symmetric:
            assert self._pin_offset
            pact_learnable_bounds = PACTLearnableClippingBounds.CLIP_LO

        elif self._qrange.is_unsigned:
            assert self._pin_offset
            pact_learnable_bounds = PACTLearnableClippingBounds.CLIP_HI

        else:
            pact_learnable_bounds = PACTLearnableClippingBounds.CLIP_LO_AND_CLIP_HI

        self._pact_learnable_bounds = pact_learnable_bounds

    def _flag_bounds_as_learnable(self):
        raise NotImplementedError

    def _update_qhparams_and_clipping_bounds(self):
        raise NotImplementedError

    def freeze(self):
        self._update_qhparams_and_clipping_bounds()
        self.clip_lo.requires_grad = False
        self.clip_hi.requires_grad = False
        self._is_frozen |= True

    def thaw(self):
        self._flag_bounds_as_learnable()
        self._is_frozen &= False

    def register_qop(self):
        self._qop = _PACTQuantiser.apply

    def call_qop(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class _PACTActivation(_PACTModule):

    def __init__(self):
        _PACTModule.__init__(self)

    def _flag_bounds_as_learnable(self):

        if self._pact_learnable_bounds == PACTLearnableClippingBounds.CLIP_LO:
            assert self._pin_offset
            self.clip_lo.requires_grad = True

        elif self._pact_learnable_bounds == PACTLearnableClippingBounds.CLIP_HI:
            assert self._pin_offset
            self.clip_hi.requires_grad = True

        else:  # self._pact_learnable_bounds == PACTLearnableClippingBounds.CLIP_LO_AND_CLIP_HI
            self.clip_lo.requires_grad = True
            self.clip_hi.requires_grad = True

    def _update_qhparams_and_clipping_bounds(self):

        if self._is_frozen:
            pass

        else:
            with torch.no_grad():

                if self._pact_learnable_bounds == PACTLearnableClippingBounds.CLIP_LO:
                    assert self._pin_offset
                    a = self.clip_lo.data
                    b = -a
                    self._check_clipping_bounds(a, b)
                    scale = get_scale(a, b, self.zero, self.n_levels, self.step)
                    self.scale.data.copy_(scale.to(device=self.scale.device))

                elif self._pact_learnable_bounds == PACTLearnableClippingBounds.CLIP_HI:
                    assert self._pin_offset
                    a = self.clip_lo.data
                    b = self.clip_hi.data
                    assert torch.all(a == 0.0)
                    self._check_clipping_bounds(a, b)
                    scale = get_scale(a, b, self.zero, self.n_levels, self.step)
                    self.scale.data.copy_(scale.to(device=self.scale.device))

                else:  # self._pact_learnable_bounds == PACTLearnableClippingBounds.CLIP_LO_AND_CLIP_HI
                    a = self.clip_lo.data
                    b = self.clip_hi.data
                    self._check_clipping_bounds(a, b)
                    if self._pin_offset:
                        scale = get_scale(a, b, self.zero, self.n_levels, self.step)
                        self.scale.data.copy_(scale.to(device=self.scale.device))
                    else:
                        zero, scale = get_zero_scale(a, b, self.n_levels, self.step)
                        self.zero.data.copy_(zero.to(device=self.scale.device))
                        self.scale.data.copy_(scale.to(device=self.scale.device))

                self._set_clipping_bounds()

    def _maybe_redirect_clip_hi_grad(self) -> Tuple[torch.Tensor, torch.Tensor]:

        if self._pact_learnable_bounds == PACTLearnableClippingBounds.CLIP_LO:
            clip_hi = _PACTRedirectClipHiGrad.apply(self.clip_lo, self.n_levels, self.step)
        else:
            clip_hi = self.clip_hi

        return self.clip_lo, clip_hi

    def call_qop(self, x: torch.Tensor) -> torch.Tensor:
        self._update_qhparams_and_clipping_bounds()
        clip_lo, clip_hi = self._maybe_redirect_clip_hi_grad()
        x = self._qop(x, clip_lo, clip_hi, self.step, self.scale)
        return x


class _PACTLinear(_PACTModule):

    def __init__(self):
        _PACTModule.__init__(self)

    def _flag_bounds_as_learnable(self):
        pass

    def _update_qhparams_and_clipping_bounds(self):

        if self._is_frozen:
            pass
        else:
            self.init_qhparams()

    def call_qop(self, x: torch.Tensor) -> torch.Tensor:
        self._update_qhparams_and_clipping_bounds()
        x = self._qop(x, self.clip_lo, self.clip_hi, self.step, self.scale)
        return x
