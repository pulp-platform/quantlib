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

import torch.fx as fx

from .applicationpoint import ApplicationPoint


class Applier(object):

    def __init__(self):
        super(Applier, self).__init__()
        self._counter: int = 0  # use `self._counter` to distinguish applications

    def _apply(self,
               g: fx.GraphModule,
               ap: ApplicationPoint,
               id_: str) -> fx.GraphModule:
        # use `id_` to annotate graph modifications
        raise NotImplementedError

    @staticmethod
    def _polish_fxgraphmodule(g: fx.GraphModule) -> None:
        """Finalise the modifications made in ``_apply``."""
        g.graph.lint()  # https://pytorch.org/docs/stable/fx.html#torch.fx.Graph.lint; this also ensures that `fx.Node`s appear in topological order
        g.recompile()   # https://pytorch.org/docs/stable/fx.html#torch.fx.GraphModule.recompile

    def apply(self,
              g:   fx.GraphModule,
              ap:  ApplicationPoint,
              id_: str) -> fx.GraphModule:

        # create a unique application identifier
        self._counter += 1
        id_ = id_ + f'_{str(self._counter)}_'

        # modify the graph
        g = self._apply(g, ap, id_)
        Applier._polish_fxgraphmodule(g)

        return g
