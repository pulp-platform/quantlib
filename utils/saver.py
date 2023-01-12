from contextlib import contextmanager
import typing
from typing import Callable, Iterable, Mapping
from torch.fx import GraphModule
from collections import OrderedDict

class Saver(object):
    def __init__(
        self,
        net : GraphModule
    ) -> None:
        self.net = net

        self._buffer_out : Mapping = OrderedDict([])
        self._hooks : Mapping = OrderedDict([])
        self._modules : Iterable = []

    def start_saving(self):

        # reinitialize all buffers
        self._buffer_out : Mapping = OrderedDict([])
        self._hooks : Mapping = OrderedDict([])
        self._modules : Iterable = []

        self._modules = list(self.net.named_modules())
        
        # define hooks
        def get_hk(n):
            def hk(module, input, output):
                self._buffer_out [n] = output
            return hk
        
        for i,(n,l) in enumerate(self._modules):
            hk = get_hk(n)
            self._hooks[n] = l.register_forward_hook(hk)

    def stop_saving(self):
        # remove hooks
        for i,(n,l) in enumerate(self._modules):
            self._hooks[n].remove()

    def get(self, n):
        return self._buffer_out[n]
    
    @contextmanager
    def saving(self):
        self.start_saving()
        try:
            yield
        finally:
            self.stop_saving()
