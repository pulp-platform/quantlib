# 
# Author(s):
# Francesco Conti <f.conti@unibo.it>
# 
# Copyright (c) 2018-2023 ETH Zurich and University of Bologna.
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
