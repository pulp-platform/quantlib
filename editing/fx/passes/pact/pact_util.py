#
# pact_util.py
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

from functools import partial
from ...util.tracing import LeafTracer, custom_symbolic_trace

from quantlib.algorithms.pact.pact_ops import *
from quantlib.algorithms.generic.generic_ops import *

#PACT operations which should not be traced through

PACT_OPS = set([_PACTActivation,
                PACTUnsignedAct,
                PACTAsymmetricAct,
                PACTConv2d,
                PACTConv1d,
                PACTConstWrap,
                PACTLinear,
                PACTSoftmax,
                PACTITAMax,
                PACTIntegerITAMax,
                PACTITAPartialMax,
                PACTIntegerITAPartialMax,
                PACTGELU,
                PACTLayerNorm,
                PACTRMSNorm,
                PACTIntegerSoftmax,
                PACTIntegerGELU,
                PACTIntegerLayerNorm,
                PACTIntegerRMSNorm,
                PACTEmbedding,
                PACTWrapModule,
                PACTHardsigmoid,
                PACTHardswish,
                PACTIntegerHardsigmoid,
                PACTIntegerHardswish,
                RequantShift,
                PACTDiv,
                PACTMean,
                PACTIntegerMean,
                PACTIntegerDiv,
                PACTExp,
                PACTIntegerExp,
                Multiply,
                ChannelwiseThreshold])

#TODO is this still reasonable??
PACT_OPS_INT = set([PACTIntegerAdd,
                    PACTIntegerConcat,
                    PACTIntegerMatmul,
                    PACTIntegerEmbedding])

#All PACT operations - ordinarily we would want to trace through the
#integerized operations
PACT_OPS_INCLUSIVE = PACT_OPS | PACT_OPS_INT

PACTTracer = LeafTracer(leaf_types=list(PACT_OPS))
PACTInclusiveTracer = LeafTracer(leaf_types=list(PACT_OPS_INCLUSIVE))

PACT_symbolic_trace = partial(custom_symbolic_trace, tracer=PACTTracer)
PACT_symbolic_trace_inclusive = partial(custom_symbolic_trace, tracer=PACTInclusiveTracer)
