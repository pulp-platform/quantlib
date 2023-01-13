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

from .modulemapping import ModuleMapping
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import NNMODULE_TO_PACTMODULE
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import PACTIdentity, PACTReLU, PACTReLU6, PACTLeakyReLU
from quantlib.algorithms.qalgorithms.qatalgorithms.pact import PACTLinear, PACTConv1d, PACTConv2d, PACTConv3d


class PTQQATRegister(OrderedDict):
    """This object is supposed to be unique (i.e., a singleton)."""

    def __setitem__(self,
                    acronym:  str,
                    fp_to_fq: ModuleMapping):  # floating-point to fake-quantised

        # validate input type
        if not isinstance(acronym, str):
            raise TypeError  # the acronym for a PTQ/QAT algorithm should be a string
        if not isinstance(fp_to_fq, ModuleMapping):
            raise TypeError

        # canonicalise
        acronym = acronym.upper()

        # validate input value
        if acronym in self.keys():
            raise ValueError  # the acronym has already be used for another PTQ/QAT algorithm

        super(PTQQATRegister, self).__setitem__(acronym, fp_to_fq)


# create indices
register = PTQQATRegister()

# -- TO BE EXPANDED BY IMPLEMENTERS OF PTQ/QAT ALGORITHMS -- #
# register PACT
register['PACT'] = NNMODULE_TO_PACTMODULE
