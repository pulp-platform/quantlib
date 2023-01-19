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

import torch

from . import ILSVRC12, common
import quantlib.editing.graphs as qg
import quantlib.backends as qb


exporter = qb.dory.DORYExporter()


# -- 1. ResNet18 -- #

# create the data set
dataset = ILSVRC12.data.get_ilsvrc12_dataset(ILSVRC12.ResNet.ILSVRC12RNTransform())
test_indices = list(range(0, 100))

# create the floating-point PyTorch network, trace it, and evaluate it
rn18 = ILSVRC12.ResNet.ResNet('ResNet18')
rn18fp = qg.fx.quantlib_symbolic_trace(root=rn18)
evrn18fp = common.evaluate_network(rn18fp, dataset, indices=test_indices)

# fake-quantise the network and evaluate it
rn18fq_uninit = common.apply_f2f_flow(rn18fp)
evrn18fq_uninit = common.evaluate_network(rn18fq_uninit, dataset, indices=test_indices)

# initialise the fake-quantisation and evaluate the network
rn18fq_init = common.initialise_quantisation(rn18fq_uninit, dataset)
evrn18fq_init = common.evaluate_network(rn18fq_init, dataset, indices=test_indices)

# true-quantise the network and evaluate it
rn18tq = common.apply_f2t_flow(rn18fq_init, dataset, ILSVRC12.ResNet.RNHeadRewriter())
evrn18tq = common.evaluate_network(rn18tq, dataset, integerise=True, indices=test_indices)

# export DORY-specific files
rn18_directory = common.get_directory('DORY', 'ILSVRC12', 'ResNet18')
x, _ = dataset.__getitem__(0)
x = x.unsqueeze(0)
x = (x / dataset.scale).floor()  # integerise
exporter.export(network=rn18tq, input_shape=x.shape, path=rn18_directory)
exporter.dump_features(network=rn18tq, x=x, path=rn18_directory)
common.zip_directory(rn18_directory)


# -- 2. MobileNetV1 -- #

# create the data set
dataset = ILSVRC12.data.get_ilsvrc12_dataset(ILSVRC12.MobileNetV1.ILSVRC12MNv1Transform())
test_indices = list(range(0, 100))

# create the floating-point PyTorch network, trace it, and evaluate it
mnv1 = ILSVRC12.MobileNetV1.MobileNetV1('standard', capacity=0.75)
mnv1_ckpts = ILSVRC12.MobileNetV1.MNv1_checkpoints()
mnv1.load_state_dict(torch.load(mnv1_ckpts[-1], map_location=torch.device('cpu')))
mnv1fp = qg.fx.quantlib_symbolic_trace(root=mnv1)
evmnv1fp = common.evaluate_network(mnv1fp, dataset, indices=test_indices)

# fake-quantise the network and evaluate it
mnv1fq_uninit = common.apply_f2f_flow(mnv1fp)
evmnv1fq_uninit = common.evaluate_network(mnv1fq_uninit, dataset, indices=test_indices)

# initialise the fake-quantisation and evaluate the network
mnv1fq_init = common.initialise_quantisation(mnv1fq_uninit, dataset)
mnv1fq_init.load_state_dict(torch.load(mnv1_ckpts[1], map_location=torch.device('cpu')))
evmnv1fq_init = common.evaluate_network(mnv1fq_init, dataset, indices=test_indices)

# true-quantise the network and evaluate it
mnv1tq = common.apply_f2t_flow(mnv1fq_init, dataset, ILSVRC12.MobileNetV1.MNv1HeadRewriter())
evmnv1tq = common.evaluate_network(mnv1tq, dataset, integerise=True, indices=test_indices)

# export DORY-specific files
mnv1_directory = common.get_directory('DORY', 'ILSVRC12', 'MobileNetV1')
x, _ = dataset.__getitem__(0)
x = x.unsqueeze(0)
x = (x / dataset.scale).floor()  # integerise
exporter.export(network=mnv1tq, input_shape=x.shape, path=mnv1_directory)
exporter.dump_features(network=mnv1tq, x=x, path=mnv1_directory)
common.zip_directory(mnv1_directory)


# -- 3. MobileNetV2 -- #

# create the data set
dataset = ILSVRC12.data.get_ilsvrc12_dataset(ILSVRC12.MobileNetV2.ILSVRC12MNv2Transform())
test_indices = list(range(0, 100))

# create the floating-point PyTorch network, trace it, and evaluate it
mnv2 = ILSVRC12.MobileNetV2.MobileNetV2('standard')
mnv2fp = qg.fx.quantlib_symbolic_trace(root=mnv2)
evmnv2fp = common.evaluate_network(mnv2fp, dataset, indices=test_indices)

# fake-quantise the network and evaluate it
mnv2fq_uninit = common.apply_f2f_flow(mnv2fp)
evmnv2fq_uninit = common.evaluate_network(mnv2fq_uninit, dataset, indices=test_indices)

# initialise the fake-quantisation and evaluate the network
mnv2fq_init = common.initialise_quantisation(mnv2fq_uninit, dataset)
evmnv2fq_init = common.evaluate_network(mnv2fq_init, dataset, indices=test_indices)

# true-quantise the network and evaluate it
mnv2tq = common.apply_f2t_flow(mnv2fq_init, dataset, ILSVRC12.MobileNetV2.MNv2HeadRewriter())
evmnv2tq = common.evaluate_network(mnv2tq, dataset, integerise=True, indices=test_indices)

# export DORY-specific files
mnv2_directory = common.get_directory('DORY', 'ILSVRC12', 'MobileNetV2')
x, _ = dataset.__getitem__(0)
x = x.unsqueeze(0)
x = (x / dataset.scale).floor()  # integerise
exporter.export(network=mnv2tq, input_shape=x.shape, path=mnv2_directory)
exporter.dump_features(network=mnv2tq, x=x, path=mnv2_directory)
common.zip_directory(mnv2_directory)
