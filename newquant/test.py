import unittest

from qrange import resolve_quantspec


class QuantSpecResolverTest(unittest.TestCase):

    def test_tuplequantspec(self):
        """Test tuple-based explicit enumerations."""

        # ordered range with step one
        range_ = (1, 2, 3, 4)
        q = resolve_quantspec(range_)
        self.assertTrue(set(range_) == set(q.range))

        # unordered range with step one
        range_ = (2, 1, 3, 4)
        q = resolve_quantspec(range_)
        self.assertTrue(set(range_) == set(q.range))

        # ordered range with step greater than one
        range_ = (-4, -2, 0, 2)
        q = resolve_quantspec(range_)
        self.assertTrue(set(range_) == set(q.range))

        # unordered range with step greater than one
        range_ = (-4, -2, 2, 0)
        q = resolve_quantspec(range_)
        self.assertTrue(set(range_) == set(q.range))

        # non-equally-spaced range
        range_ = (1, 2, 3, 5)
        self.assertRaises(ValueError, lambda: resolve_quantspec(range_))  # Why am I using a lambda expression here? See here: https://stackoverflow.com/a/6103930

    def test_dictquantspec(self):
        """Test dictionary-based (compact) specifications."""

        # number of levels, unsupported key
        dict_ = {'n_levels': 16, 'offset': -2, 'step': 2}
        self.assertRaises(ValueError, lambda: resolve_quantspec(dict_))

        # number of levels, offset
        dict_ = {'n_levels': 16, 'offset': -2}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(-2, -2 + 1 * 16, 1)) == set(q.range))

        # even number of levels, signedness (sub-range of signed int)
        dict_ = {'n_levels': 16, 'signed': True}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(-8, -8 + 16, 1)) == set(q.range))

        # odd number of levels, signedness (sub-range of signed "limp" int)
        dict_ = {'n_levels': 15, 'signed': True}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(-7, -7 + 15, 1)) == set(q.range))

        # even number of levels, signedness (sub-range of unsigned int)
        dict_ = {'n_levels': 16, 'signed': False}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(0, 16, 1)) == set(q.range))

        # odd number of levels, signedness (sub-range of unsigned "limp" int)
        dict_ = {'n_levels': 15, 'signed': False}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(0, 15, 1)) == set(q.range))

        # bitwidth, unsupported key
        dict_ = {'bitwdith': 4, 'offset': -10, 'step': 4}
        self.assertRaises(ValueError, lambda: resolve_quantspec(dict_))

        # bitwidth, offset
        dict_ = {'bitwidth': 4, 'offset': -10}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(-10, -10 + 1 * 2 ** 4, 1)) == set(q.range))

        # bitwidth, signedness (signed int)
        dict_ = {'bitwidth': 4, 'signed': True}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(-2 ** (4 - 1), 2 ** (4 - 1), 1)) == set(q.range))

        # bitwidth, signedness (unsigned int)
        dict_ = {'bitwidth': 4, 'signed': False}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(0, 2 ** 4, 1)) == set(q.range))

        # limp bitwidth, unsupported key
        dict_ = {'limpbitwidth': 3, 'offset': 5, 'step': 3}
        self.assertRaises(ValueError, lambda: resolve_quantspec(dict_))

        # limp bitwidth, offset
        dict_ = {'limpbitwidth': 3, 'offset': 5}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(5, 5 + 2 ** 3 - 1, 1)) == set(q.range))

        # limp bitwidth, signedness (signed "limp" int)
        dict_ = {'limpbitwidth': 3, 'signed': True}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(- 2 ** (3 - 1) + 1, 2 ** (3 - 1), 1)) == set(q.range))

        # limp bitwidth, signedness (unsigned "limp" int)
        dict_ = {'limpbitwidth': 3, 'signed': False}
        q = resolve_quantspec(dict_)
        self.assertTrue(set(range(0, 2 ** 3 - 1, 1)) == set(q.range))

    def test_strquantspec(self):
        """Test the string-based (syntactic sugar) specifications."""

        # binary range
        q = resolve_quantspec('binary')
        self.assertTrue({-1, 1} == set(q.range))

        # ternary range
        q = resolve_quantspec('ternary')
        self.assertTrue({-1, 0, 1} == set(q.range))

        # non-supported string
        self.assertRaises(ValueError, lambda: resolve_quantspec('quaternary'))
