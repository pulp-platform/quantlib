import enum


@enum.unique
class QGranularity(enum.Enum):
    PER_LAYER = 0
    PER_CHANNEL = 1
