#
# general.py
#
# Author(s):
# Georg Rutishauser <georgr@iis.ee.ethz.ch>
# Moritz Scherer <scheremo@iis.ee.ethz.ch>
#
# Copyright (c) 2020-2021 ETH Zurich.
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

from typing import Union, Optional, Tuple, List

import torch
from torch import nn, fx
from torch.fx.passes.shape_prop import ShapeProp
from torch.nn import functional as F
from .pass_base import FxPass, SequentialPass, ModifySequentialPatternPass, ModularizePass
from ..util import module_of_node, get_qualified_prefix

__all__ = [
    'MergeConvBNPass', 'ModularizeActivationsPass', 'RetracePass',
    'InsertModuleAfterNodePass', 'InsertModuleBetweenNodesPass',
    'InsertModuleBetweenModulesPass', 'ShapePropPass'
]


def merge_conv_bn_fun(ml: list):
    assert len(ml) == 2, "List passed to merge_conv_bn_fun should have length 2"
    conv_module = [
        m for m in ml if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Conv3d))
    ][0]
    bn_module = [
        m for m in ml
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    ][0]
    if conv_module.bias is None:
        return
    bias_data = conv_module.bias.data.clone().detach()
    conv_module.bias = None
    bn_module.running_mean.data -= bias_data


class MergeConvBNPass(SequentialPass):

    def __init__(self, trace: callable = fx.symbolic_trace):
        passes = []
        for conv, bn, dim in [(nn.Conv1d, nn.BatchNorm1d, "1D"),
                              (nn.Conv2d, nn.BatchNorm2d, "2D"),
                              (nn.Conv3d, nn.BatchNorm3d, "3D")]:
            pattern = nn.Sequential(conv(1, 1, 1), bn(1))
            passes.append(
                ModifySequentialPatternPass(pattern, trace, merge_conv_bn_fun,
                                            f"_MERGE_CONV_BN_{dim}"))
        super(MergeConvBNPass, self).__init__(*passes,
                                              name_prefix="_MERGE_CONV_BN_PASS")


class ModularizeActivationsPass(ModularizePass):

    inplace_act_functions = (F.threshold_, F.relu_, F.hardtanh_, F.elu_,
                             F.leaky_relu_, F.rrelu_)

    act_function_to_module = {
        F.threshold:
            nn.Threshold,
        F.threshold_:
            nn.Threshold,
        F.relu:
            nn.ReLU,
        F.relu_:
            nn.ReLU,
        F.hardtanh:
            nn.Hardtanh,
        F.hardswish:
            nn.Hardswish,
        F.relu6:
            nn.ReLU6,
        F.elu:
            nn.ELU,
        F.elu_:
            nn.ELU,
        F.selu:
            nn.SELU,
        F.celu:
            nn.CELU,
        F.leaky_relu:
            nn.LeakyReLU,
        F.leaky_relu_:
            nn.LeakyReLU,
        F.prelu:
            nn.PReLU,
        F.rrelu:
            nn.RReLU,
        F.rrelu_:
            nn.RReLU,
        F.glu:
            nn.GLU,
        F.gelu:
            nn.GELU,
        F.logsigmoid:
            nn.LogSigmoid,
        F.hardshrink:
            nn.Hardshrink,
        F.tanhshrink:
            nn.Tanhshrink,
        F.softsign:
            nn.Softsign,
        F.softplus:
            nn.Softplus,
        F.softmin:
            nn.Softmin,
        F.softmax:
            nn.Softmax,
        F.softshrink:
            nn.Softshrink,
        #F.log_softmax : nn.LogSoftmax # interfaces don't
        #conform as they should for logSoftmax...
        F.tanh:
            nn.Tanh,
        F.sigmoid:
            nn.Sigmoid,
        F.hardsigmoid:
            nn.Hardsigmoid,
        F.silu:
            nn.SiLU,
        F.mish:
            nn.Mish
    }

    @staticmethod
    def act_node_to_module(node):
        module_inst_args = node.args[1:]
        module_inst_kwargs = {
            k: v for k, v in node.kwargs.items() if k != 'input'
        }
        if node.target in inplace_act_functions:
            module_inst_kwargs['inplace'] = True
        module_call_args = node.args[0:1]
        module_call_kwargs = {
            k: v for k, v in node.kwargs.items() if k == 'input'
        }
        module_class = self.act_function_to_module[node.target]
        return (module_class(*module_inst_args, **module_inst_kwargs),
                module_call_args, module_call_kwargs)

    def __init__(self):
        super(ModularizeActivationsPass, self).__init__(
            op='call_function',
            target=tuple(k for k in act_function_to_module.keys()),
            name="MODULARIZE_ACTIVATIONS_PASS")


class RetracePass(FxPass):

    def __init__(self, trace: callable):
        super(RetracePass, self).__init__()
        self.trace = trace

    def run_pass(self, gm: fx.GraphModule):
        return self.trace(gm)


class InsertModuleAfterNodePass(FxPass):

    def __init__(self,
                 node_before: fx.Node,
                 module: nn.Module,
                 name: str,
                 insert_target: Optional[str] = None):
        super(InsertModuleAfterNodePass, self).__init__()
        self.node_before = node_before
        self.module = module
        self.name = name
        self.insert_target = insert_target

    def run_pass(self, gm: fx.GraphModule):
        target_list = []
        if self.insert_target is None and self.node_before.op == 'call_module':
            before_prefix = get_qualified_prefix(self.node_before.target)
            if len(before_prefix):
                target_list.append(before_prefix)
        elif self.insert_target is not None:
            target_list.append(self.insert_target)
        target_list.append(f"_QL_INSERTED_AFTER_{self.name.upper()}")
        new_target = '.'.join(target_list)
        gm.add_submodule(new_target, self.module)

        # need to keep track of users before adding the new node
        users_list = [u for u in self.node_before.users]

        with gm.graph.inserting_after(self.node_before):
            new_node = gm.graph.call_module(new_target,
                                            args=(self.node_before,))

        for u in users_list:
            u.replace_input_with(self.node_before, new_node)

        return gm


class InsertModuleBetweenNodesPass(FxPass):

    def __init__(self,
                 node_before: fx.Node,
                 node_after: fx.Node,
                 module: nn.Module,
                 name: str,
                 insert_target: Optional[str] = None):
        super(InsertModuleBetweenNodesPass, self).__init__()
        assert node_after in [
            u for u in node_before.users
        ], f"Supplied invalid node pair to InsertModuleBetweenNodesPass: Node {node_after} is not a user of {node_before}!"
        self.node_before = node_before
        self.node_after = node_after
        self.module = module
        self.name = name
        self.insert_target = insert_target

    def run_pass(self, gm: fx.GraphModule):
        target_list = []
        if self.insert_target is None and self.node_before.op == 'call_module':
            before_prefix = get_qualified_prefix(self.node_before.target)
            if len(before_prefix):
                target_list.append(before_prefix)
        elif self.insert_target is not None:
            target_list.append(self.insert_target)
        target_list.append(f"_QL_INSERTED_BETWEEN_{self.name.upper()}")
        new_target = '.'.join(target_list)
        gm.add_submodule(new_target, self.module)

        with gm.graph.inserting_after(self.node_before):
            new_node = gm.graph.call_module(new_target,
                                            args=(self.node_before,))

        self.node_after.replace_input_with(self.node_before, new_node)

        return gm


class InsertModuleBetweenModulesPass(SequentialPass):

    def __init__(self,
                 modules_before: Union[list, tuple],
                 modules_after: Union[list, tuple],
                 make_module_fn: callable,
                 name: str,
                 combine: Union['force', 'conservative', 'none'] = 'none'):
        # the combine flag specifies what to do with node sequences which match
        # the pattern but where the first node has multiple users:
        # - 'force' will insert a single replacement module after the before_node and before
        # all users, regardless of how many users match the pattern, so long as
        # one user matches
        # - 'conservative' will insert a single replacement module after the before_node
        #  and before all users if all users match the pattern, otherwise it
        # will insert a separate module between the before_node and each of the
        # matching users
        # - 'none' will insert separate module between the before_node and the
        # matching users regardless of the matching status of the users.
        super(InsertModuleBetweenModulesPass, self).__init__(name_prefix=name)
        self.modules_before = tuple(modules_before)
        self.modules_after = tuple(modules_after)
        self.make_module_fn = make_module_fn
        self.name = name
        self.combine = combine

    def retarget(self, gm: fx.GraphModule):
        for k in self.named_subpasses().keys():
            self.remove_subpass(k)

        passes = []
        idx = 0
        for node in gm.graph.nodes:
            if node.op == 'call_module':
                m = module_of_node(gm, node)
                if isinstance(m, self.modules_before):
                    insert_before_users = []
                    for u in node.users:
                        if u.op == 'call_module':
                            user_module = module_of_node(gm, u)
                            if isinstance(user_module, self.modules_after):
                                insert_before_users.append((u, user_module))
                    if len(insert_before_users) >= 1 and (
                        (len(insert_before_users) == len(node.users)
                         and self.combine == 'conservative')
                            or self.combine == 'force'):
                        new_module = self.make_module_fn(
                            m, insert_before_users[0][1])
                        passes.append(
                            InsertModuleAfterNodePass(
                                node, new_module, f"{self.name.upper()}_{idx}"))
                        idx += 1
                    else:
                        for u, user_mod in insert_before_users:
                            new_module = self.make_module_fn(m, user_mod)
                            passes.append(
                                InsertModuleBetweenNodesPass(
                                    node, u, new_module,
                                    f"{self.name.upper()}_{idx}"))
                            idx += 1

        super(InsertModuleBetweenModulesPass, self).setup_passes(passes)


class ShapePropPass(FxPass):
    # a wrapper for the shape propagation pass of torch.fx
    def __init__(self,
                 *shapes_in: Union[Tuple[int], List[int], torch.Size],
                 dtype_in: torch.dtype = torch.float32):
        super(ShapePropPass, self).__init__()
        self.shapes_in = [torch.Size(s) for s in shapes_in]
        self.dtype_in = dtype_in

    def run_pass(self, gm: fx.GraphModule):
        training = gm.training
        gm.eval()
        sp = ShapeProp(gm)
        inp = [torch.rand(s, dtype=self.dtype_in) for s in self.shapes_in]
        sp.run(*inp)
        if training:
            # you really shouldn't be passing over GraphModules in non-eval
            # state, but you do you!
            gm.train()
        return gm
