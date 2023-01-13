# 
# Author(s):
# Matteo Spallanzani <spmatteo@iis.ee.ethz.ch>
# 
# Copyright (c) 2020-2022 ETH Zurich and University of Bologna.
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

import random
import torch
import torch.fx as fx
from typing import Optional

from quantlib.editing.editing.float2fake import F2F8bitPACTConverter
from quantlib.editing.editing.fake2true import F2T24bitConverter
from quantlib.editing.editing.editors.base.editor import Editor


def apply_f2f_flow(gmfp: fx.GraphModule) -> fx.GraphModule:
    """Pass a floating-point PyTorch network through QuantLib conversions:
    * float-to-fake (F2F) conversion;
    * uninitialised fake-quantised to initialised fake-quantised;
    * fake-to-true (F2T) conversion.
    """

    gmfp.eval()

    f2fconverter = F2F8bitPACTConverter()
    gmfq_uninit = f2fconverter(gmfp)

    return gmfq_uninit


def initialise_quantisation(gmfq_uninit:     fx.GraphModule,
                            dataset:         torch.utils.data.Dataset,
                            warmup_fraction: float = 0.0002) -> fx.GraphModule:
    """Bring a fake-quantised network from the uninitialised to the
    initialised state.

    This function assumes that the network has a single input named ``x``.

    """

    # import dependencies used only within this function's scope
    from quantlib.algorithms.qmodules.qmodules.qmodules import _QModule
    from quantlib.editing.graphs.nn import HarmonisedAdd

    # validate the `num_iter` argument
    num_iter = int(len(dataset) * warmup_fraction)
    if not (num_iter > 0):
        raise ValueError  # `num_iter` must be positive, so maybe you want to increase `warmup_fraction`

    # bring `_QModule`s in the "observing" state
    for name, module in gmfq_uninit.named_modules():
        if isinstance(module, (_QModule, HarmonisedAdd)):
            module.start_observing()

    # collect statistics
    indices = list(range(0, len(dataset)))
    random.shuffle(indices)
    for i in range(0, num_iter):
        x, _ = dataset.__getitem__(indices[i])
        x = x.unsqueeze(0)
        _ = gmfq_uninit(x)  # statistics are collected by `TensorObserver`s

    # use the collected statistics to initialise the quantisers' hyper-parameters
    for n, m in gmfq_uninit.named_modules():
        if isinstance(m, _QModule):
            m.stop_observing()

    # now the fake-quantised network is in the "initialised" state
    gmfq_init = gmfq_uninit

    return gmfq_init


def apply_f2t_flow(gmfq_init:     fx.GraphModule,
                   dataset:       torch.utils.data.Dataset,
                   custom_editor: Optional[Editor] = None) -> fx.GraphModule:

    if not hasattr(dataset, 'scale'):
        raise ValueError

    # true-quantise the network
    f2tconverter = F2T24bitConverter(custom_editor=custom_editor)
    x, y = dataset.__getitem__(0)
    gmtq = f2tconverter(gmfq_init, {'x': {'shape': x.unsqueeze(0).shape, 'scale': dataset.scale}})

    return gmtq
