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

from collections import OrderedDict
import torch.fx as fx

from quantlib.editing.editing.editors.base import ApplicationPoint


class NodesMap(ApplicationPoint, OrderedDict):
    """The ``ApplicationPoint`` for pattern-matching-based rewriting rules."""

    def __setitem__(self, pn: fx.Node, dn: fx.Node):
        """A ``NodesMap`` can only map ``fx.Node``s to ``fx.Node``s."""
        if not (isinstance(pn, fx.Node) and isinstance(dn, fx.Node)):
            raise TypeError
        super(ApplicationPoint, self).__setitem__(pn, dn)
