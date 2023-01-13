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

"""Define abstractions to simplify the handling of ``fx.Node`` objects.

The first functionality is being able to partition ``fx.Node``s according to
coarser-grained taxonomies than the basic one provided by "opcodes". This
functionality is useful in at least two cases:

  * when dealing with rewriting rules involving non-trivial query graphs, it
    is useful to have a shortcut to identify I/O nodes (i.e., ``placeholder``s
    and ``output``s);
  * when dealing with rewriting rules replacing non-modular API calls with
    modular ones, it is useful to have a shortcut to identify such non-modular
    API calls.

The second functionality is being able to "unpack" the arguments passed to
``fx.Node``s. This functionality is useful in at least one case:

  * when creating modular API objects from corresponding non-modular ones
    (e.g., to copy the keyword arguments passed to the non-modular object into
    the corresponding ``nn.Module``).

The third functionality is specific to ``fx.Node``s representing traced
``nn.Module``s, and consists in being able to retrieve the ``nn.Module``
object associated with a given ``fx.Node`` of opcode ``call_module``.

"""

from enum import Enum, unique
import torch
import torch.fx as fx
from typing import Tuple, Dict, NamedTuple, Union, Any


# 1. IDENTIFY FX.NODE OPCODES

# PLHD  OUT_  ATTR  CMOD  CFUN  CMET #
#  |     |     |     |     |     |   #
#   \   /      |     |      \   /    #
#   I/O_      ATTR  CMOD    CNMO     #
#    |         |     |       |       #
#     \       /       \     /        #
#       \   /           \ /          #
#       DATA            CALL         #
#         |              |           #
#          \            /            #
#            \        /              #
#              \    /                #
#               ALL_                 #

# singleton opcode classes
FXOPCODE_PLACEHOLDER   = {'placeholder'}
FXOPCODE_OUTPUT        = {'output'}
FXOPCODE_GET_ATTR      = {'get_attr'}
FXOPCODE_CALL_MODULE   = {'call_module'}
FXOPCODE_CALL_FUNCTION = {'call_function'}
FXOPCODE_CALL_METHOD   = {'call_method'}

# higher-level opcode classes
FXOPCODES_IO              = FXOPCODE_PLACEHOLDER   | FXOPCODE_OUTPUT
FXOPCODES_DATA            = FXOPCODE_GET_ATTR      | FXOPCODES_IO
FXOPCODES_CALL_NONMODULAR = FXOPCODE_CALL_FUNCTION | FXOPCODE_CALL_METHOD
FXOPCODES_CALL            = FXOPCODE_CALL_MODULE   | FXOPCODES_CALL_NONMODULAR
FXOPCODES_ALL             = FXOPCODES_DATA         | FXOPCODES_CALL


@unique
class FXOpcodeClasses(Enum):
    PLACEHOLDER     = FXOPCODE_PLACEHOLDER
    OUTPUT          = FXOPCODE_OUTPUT
    GET_ATTR        = FXOPCODE_GET_ATTR
    CALL_MODULE     = FXOPCODE_CALL_MODULE
    CALL_FUNCTION   = FXOPCODE_CALL_FUNCTION
    CALL_METHOD     = FXOPCODE_CALL_METHOD
    IO              = FXOPCODES_IO
    DATA            = FXOPCODES_DATA
    CALL_NONMODULAR = FXOPCODES_CALL_NONMODULAR
    CALL            = FXOPCODES_CALL
    ALL             = FXOPCODES_ALL


# 2. EXTRACT FX.NODE ARGUMENTS

FxNodeArgType = Union[Tuple[Any, ...], list, Dict[str, Any], slice, fx.Node, str, int, float, bool, torch.dtype, torch.Tensor, torch.device, None]


class IDdFxNodeArg(NamedTuple):
    """Attach a numeric identifier to an ``fx.Node``, signalling its position
    in the signature of its user ``fx.Node``.
    """
    id:     int
    fxnode: fx.Node


def unpack_then_split_fxnode_arguments(n: fx.Node) -> Tuple[Tuple[IDdFxNodeArg, ...], Tuple[FxNodeArgType, ...], Dict[str, fx.Node], Dict[str, FxNodeArgType]]:
    """Retrieve and partition the inputs to an ``fx.Node``.

    Each ``fx.Node`` represents the execution of an operation in the flow of
    the target network. Each operation is implemented as a Python function.
    The signature of a Python function can be modelled as an immutable
    :math:`N`-tuple of the form

    .. math:
       ((1, m_{1}, t_{1}), \dots, (N, m_{N}, t_{N})) \,,

    where each component is associated with an argument and is itself a tuple.
    Each such tuple has three components:
    * an integer encoding the position of the argument in the signature;
    * a key-value pair, where the key is a string in the collection
      :math:`\{ "positional", "default", "arbitrary", "keyword" \}` and the
      value is a string representing the symbolic name of the argument;
    * a Python type representing the type of the argument.

    Due to the logic of the Python interpreter, it is possible to pass
    arguments out of order. Therefore, we have a distinction between the
    "runtime signature" with which a Python function is called, and the
    "standard signature" which is the immutable entity described above.
    Resolving a Python runtime signature amounts to mapping each positional
    and keyword input of the target ``fx.Node`` to the unique integer index
    encoding its position in the standard signaure of the ``fx.Node``.

    TODO: at the moment, we do not support this disambiguation; however, I
          allow for future extensibility by noting the position in which each
          positional argument appears in the (symbolic) call.
    """

    # split positional from keyword arguments
    args, kwargs = n.args, n.kwargs

    # split `fx.Node` from non-`fx.Node` arguments
    # positional
    fxnode_args, other_args = [], []
    for i, a in enumerate(args):
        (fxnode_args if isinstance(a, fx.Node) else other_args).append(IDdFxNodeArg(id=i, fxnode=a))  # https://stackoverflow.com/a/12135169
    fxnode_args = tuple(fxnode_args)  # freeze container objects
    other_args  = tuple(other_args)
    # keyword
    fxnode_kwargs, other_kwargs = {}, {}
    for k, v in kwargs.items():
        (fxnode_kwargs if isinstance(v, fx.Node) else other_kwargs)[k] = v

    return fxnode_args, other_args, fxnode_kwargs, other_kwargs
