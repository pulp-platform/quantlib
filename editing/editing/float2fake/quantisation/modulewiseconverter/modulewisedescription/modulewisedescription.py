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

"""We build ``ModuleWiseFinder``s and ``ModuleWiseReplacer``s around this data
structure.
"""

from collections import OrderedDict
from typing import NamedTuple, Tuple

from ..applicationpoint import PartitionId
from .nametomodule import N2MFilter, N2MFilterSpecType, resolve_n2mfilterspec
from ...qdescription import QDescription, QDescriptionSpecType, resolve_qdescriptionspec


# -- DESCRIPTION OF A SINGLE PARTITION -- #

class Partition(NamedTuple):
    n2mfilter:    N2MFilter
    qdescription: QDescription


# -- DESCRIPTION OF A COLLECTION OF PARTITIONS -- #

PartitionSpecType             = Tuple[N2MFilterSpecType, QDescriptionSpecType]
ModuleWiseDescriptionSpecType = Tuple[PartitionSpecType, ...]


class ModuleWiseDescription(OrderedDict):

    def __init__(self, modulewisedescriptionspec: ModuleWiseDescriptionSpecType):

        super(ModuleWiseDescription, self).__init__()

        # validate input value
        if not all((isinstance(item_, tuple) and (len(item_) == 2)) for item_ in modulewisedescriptionspec):
            raise ValueError  # ill-formed description

        for i, (n2mfilterspec, qdescriptionspec) in enumerate(modulewisedescriptionspec):
            # resolve the `PartitionSpec`
            n2mfilter = resolve_n2mfilterspec(n2mfilterspec)
            qdescription = resolve_qdescriptionspec(qdescriptionspec)
            # create and register the `Partition`
            partition = Partition(n2mfilter, qdescription)
            self.__setitem__(PartitionId(i), partition)

    def __setitem__(self, id_: PartitionId, partition: Partition):

        if not isinstance(id_, PartitionId):
            raise TypeError
        if not isinstance(partition, Partition):
            raise TypeError

        super(ModuleWiseDescription, self).__setitem__(id_, partition)
