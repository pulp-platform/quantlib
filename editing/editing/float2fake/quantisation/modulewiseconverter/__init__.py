"""This package implements the abstraction to perform module-wise
float-to-fake conversions.

This ``Rewriter``, named ``ModuleWiseReplacer``, is built around
``ModuleWiseDescription``s. A ``ModuleWiseDescription is a map from
``PartitionId``s to pairs coupling an ``N2MFilter`` with a ``QDescription``.

At search time, a ``ModuleWiseFinder`` builds a ``NameToModule`` object
mapping each ``nn.Module`` in the target ``fx.GraphModule`` to its symbolic
name; then, it uses ``N2MFilter``s to map each the ``fx.Node``s associated
with an ``nn.Module`` to a partition. Each of this pairs is modelled as a
``NodeWithPartition`` data structure.

``NodeWithPartition`` is the ``ApplicationPoint`` data structure used to
exchange information between ``ModuleWiseFinder``s and their associated
``ModuleWiseReplacer``s.

At rewriting time, for each received ``NodeWithPartition``, a
``ModuleWiseReplacer`` looks up the ``PartitionId`` and retrieves the
``QDescription`` associated with the partition; then, it creates a
``_QModule`` to replace the original floating-point one; finally, it replaces
the floating-point module with the quantised one.

Users can specify a ``ModuleWiseDescription`` as a tuple of pairs. Each pair
describes a partition, and will be assigned a unique ``PartitionId``. The
first element in each pair should be an object of type ``N2MFilterSpecType``
describing how to filter the namespace of the target ``fx.GraphModule``; the
second element in each pair should be a ``QDescriptionSpecType`` object
describing how to quantise each floating-point module.

"""

from .rewriter import ModuleWiseConverter

__all__ = [
    'ModuleWiseConverter',
]
