"""This package implements the abstractions do described quantised modules.

Apart from the canonical collection data structures, the module defines
several syntactic constructs to specify such structures, and implements
canonicalisation functions to convert non-canonical descriptions into
canonical ones.

"""

#      ((`acronym` -> ModuleMapping) + `kwargs`) = QAlgorithm                  #
#                                                     |                        #
#  (QGranularity + QRange + QHParamsInitStrategy + QAlgorithm) = QDescription  #

from .qdescription import QDescription, QDescriptionSpecType, resolve_qdescriptionspec
