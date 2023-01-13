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

from functools import partial


_QUANTLIB_LOG_HEADER = "[QuantLib] "
_QUANTLIB_WNG_HEADER = "[QuantLib warning] "
_QUANTLIB_ERR_HEADER = "[QuantLib error] "


def quantlib_msg_header(header: str, obj_name: str = "") -> str:
    """Create a header for QuantLib log or error messages.

    Arguments:
        obj_name: the (optional) name of the function or object class that
            is triggering the logging or error.

    """
    return header + (f"{obj_name}: " if obj_name != "" else obj_name)


quantlib_log_header = partial(quantlib_msg_header, header=_QUANTLIB_LOG_HEADER)
quantlib_wng_header = partial(quantlib_msg_header, header=_QUANTLIB_WNG_HEADER)
quantlib_err_header = partial(quantlib_msg_header, header=_QUANTLIB_ERR_HEADER)
