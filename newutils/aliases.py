from typing import NewType


# for information that is not yet available
UnknownType = NewType('UnknownType', type(None))
UNKNOWN = UnknownType(None)
