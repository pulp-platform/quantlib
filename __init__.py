# 
# __init__.py
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

"""QuantLib: a package to quantise deep neural networks.

The QuantLib package partitions its abstractions into three namespaces:
* the ``algorithms`` sub-package implements the extensions to PyTorch's ``nn``
  namespace required to build networks supporting **quantisation-aware
  training (QAT)**;
* the ``editing`` sub-package implements the abstractions required to
  transform floating-point deep neural networks into fake-quantised networks,
  and fake-quantised networks into true-quantised (i.e., integerised) ones;
* the ``backend`` sub-package implements the abstractions required to export
  trained and integerised quantised neural networks (QNNs) to ONNX formats
  compatible with different platforms.

There is also a ``utils`` sub-package, but since it is meant for developers I
do not include in the canonical QuantLib triad.

"""
