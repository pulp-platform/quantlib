import unittest

from .qgranularity import resolve_qgranularityspec


class QGranularitySolverTest(unittest.TestCase):

    def test_tuple_qgranularityspec(self):
        """Test the string-based"""

        # incorrect specification (index dimensions must be positive)
        reference = (-1, 0, 3)
        tuple_ = reference
        self.assertRaises(ValueError, lambda: resolve_qgranularityspec(tuple_))

        # correct specification
        reference = (0, 2)
        tuple_ = reference
        qgranularity = resolve_qgranularityspec(tuple_)
        self.assertTrue(reference == qgranularity)

    def test_str_qgranularityspec(self):
        """Test the string-based (syntactic sugar) specifications."""

        # per-array (AKA per-tensor)
        reference = tuple()
        qgranularity = resolve_qgranularityspec('per-array')
        self.assertTrue(reference == qgranularity)

        # per-channel (for weights)
        reference = (0,)
        qgranularity = resolve_qgranularityspec('per-outchannel_weights')
        self.assertTrue(reference == qgranularity)

        # non-supported string
        self.assertRaises(ValueError, lambda: resolve_qgranularityspec('along_batch-size_features'))
