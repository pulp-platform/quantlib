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
from typing import List

from .editor import Editor


class ComposedEditor(Editor):
    """``Editor`` applying a sequence of editing steps to the target graph."""

    def __init__(self, children_editors: List[Editor]):

        # validate input
        if not (isinstance(children_editors, list) and all(map(lambda editor: isinstance(editor, Editor), children_editors))):
            raise TypeError

        super(ComposedEditor, self).__init__()
        self._children_editors = children_editors

    def apply(self, g: fx.GraphModule, *args, **kwargs) -> fx.GraphModule:

        for editor in self._children_editors:
            g = editor(g, *args, **kwargs)

        return g
