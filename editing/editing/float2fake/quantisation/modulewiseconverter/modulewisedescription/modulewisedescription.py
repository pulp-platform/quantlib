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
