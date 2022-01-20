from typing import Tuple, Dict
from typing import Union


class QRange(object):
    """A class representing the integer set where an array can take values.

    I make the fundamental assumption that the fake-quantised operands of a
    fake-quantised neural network can be mapped to a range of equally-spaced
    integers by simply dividing by the quantum (i.e., the floating-point
    number associated with the fake-quantised array).

    Therefore, when building fake-quantised ``nn.Module``s, I assume that the
    user will need to specify such integer ranges, which will be attached to
    the target fake-quantised ``torch.Tensor`` (the weights in case of linear
    operations and the features in case of activation operations, which might
    be the identity or non-linear operations).

    """
    def __init__(self, n_levels: int, offset: int, step: int):
        self._n_levels = n_levels
        self._offset = offset
        self._step = step

    @property
    def range(self) -> Tuple[int, ...]:
        return tuple(range(self._offset, self._offset + self._step * self._n_levels, self._step))

    @property
    def n_levels(self) -> int:
        return self._n_levels

    @property
    def offset(self) -> int:
        return self._offset

    @property
    def step(self) -> int:
        return self._step

    @property
    def min(self) -> int:
        return self.range[0]

    @property
    def max(self) -> int:
        return self.range[-1]


# I define a QuantSpec as the union data type `QuantSpec := Union[Tuple[int, ...], Dict[str, int], str]`.

def resolve_quantspec(quantspec: Union[Tuple[int, ...], Dict[str, int], str]) -> QRange:
    """A function to disambiguate user specifications of integer ranges.

    During my four-year experience in research on quantised neural networks I
    have come across of several ways to describe integer ranges.

    * The UINTX and INTX integer ranges represent integers using the two's
      complement representation and can be specified via the positive integer
      bitwidth and the signedness (a Boolean);
    * The "limp" UINTX and INTX integer ranges, which represent integers
      without the two's complement representation and therefore can represent
      one value less than their standard counterparts (zero is signed in this
      representation). The highest value is lost in the case of UINTX (e.g.,
      255 with 8 bits), and the lowest value is lost in the case of INTX
      (e.g., -128 with 8 bits). In particular, the ternary range :math:`\{ -1,
      0, 1 \}` can be represented as "limp" INT2.
    * Explicit enumerations, where the composing values are equally spaced;
      these ranges can be expressed in more compact form by taking their
      initial values (the offset), a positive integer step, and the number of
      components that they have; an example of such a range is the binary one,
      :math:`\{ -1, 1 \}`, which has two levels, an offset of -1 and a step of
      two.

    The integer ranges specified by UINTX, INTX, "limp" UINTX and "limp" INTX
    are just a subset of all the possible ranges of equally-spaced integers.
    However, many spaced-by-one ranges of integers can be immersed into a
    digital integer range; for instance, the range :math:`\{ -1, 0, 1, 2 \}`
    is a subset of :math:`\{ -8, -7, \dots, 7 \}`, which is the INT4 range.

    In QuantLib, I allow for several ways of specifying integer ranges.

    * Explicit enumerations in the form of tuples of integers: such tuples can
      be passed not sorted, but once sorted their components must be
      equally-spaced. This format must be used to express ranges with
      spacing steps greater than one.
    * Compact dictionary-based specifications. I allow for several formats:
      * by specifying the number of levels (a positive integer greater than
        one) and an integer offset;
      * by specifying the number of levels (a positive integer greater than
        one) and the signedness (a Boolean);
      * by specifying the bitwidth (a positive integer) and and integer
        offset;
      * by specifying the bitwidth (a positive integer) and the signedness (a
        Boolean); this is the suggested way to specify UINTX and INTX formats;
      * by specifying the "limp" bitwidth (a positive integer greater than
        one) and an integer offset;
      * by specifying the "limp" bitwidth (a positive integer greater than
        one) and the signedness (a Boolean); this is the suggested way to
        specify "limp" UINTX and "limp" INTX formats.
      The first sub-format is a superset of the second; the third is a
      superset of the fourth; the fifth is a superset of the sixth.
      The first sub-format is a superset of both the third and the fifth:
      when the number of levels is an exact power-of-two, the specification is
      equivalent to the third one; when it is a power-of-two minus one, it is
      equivalent to the fifth one. The second sub-format is a superset of both
      the fourth and the sixth: when the number of levels is an exact
      power-of-two, the specification is equivalent to the fourth one; when it
      is a power-of-two minus one, it is equivalent to the sixth one.
    * String-based specifications for two particular formats: ``binary``
      (which creates the range :math:`\{ -1, 1 \}`) and ``ternary`` (which
      creates the range :math:`\{ -1, 0, 1 \}`.

    """

    # tuple spec? (explicit, sorted integer range)
    if isinstance(quantspec, tuple):
        quantspec = sorted(quantspec)
        step = set([j - i for i, j in zip(quantspec[:-1], quantspec[1:])])
        if len(step) > 1:
            raise ValueError("[QuantLib] Tuple QuantSpecs should be composed of equally-spaced integers.")
        else:
            step = step.pop()
        offset = min(quantspec)
        n_levels = len(quantspec)

    # dict spec?
    elif isinstance(quantspec, dict):

        # the keys of the dictionary define the dictionary specification format
        quantspec_keys = set(quantspec.keys())

        # when using the dictionary specification format, the step is implicitly set to one
        step = 1

        # switch depending on supported dictionary specification formats
        if quantspec_keys == {'n_levels', 'offset'}:
            offset = quantspec['offset']
            n_levels = quantspec['n_levels']

        elif quantspec_keys == {'n_levels', 'signed'}:
            n_levels = quantspec['n_levels']
            offset = -((n_levels - 1) // 2 if n_levels % 2 == 1 else n_levels // 2) if quantspec['signed'] else 0

        elif quantspec_keys == {'bitwidth', 'offset'}:
            offset = quantspec['offset']
            n_levels = 2 ** quantspec['bitwidth']

        elif quantspec_keys == {'bitwidth', 'signed'}:
            offset = - 2 ** (quantspec['bitwidth'] - 1) if quantspec['signed'] else 0
            n_levels = 2 ** quantspec['bitwidth']

        elif quantspec_keys == {'limpbitwidth', 'offset'}:
            offset = quantspec['offset']
            n_levels = 2 ** quantspec['limpbitwidth'] - 1

        elif quantspec_keys == {'limpbitwidth', 'signed'}:
            offset = (- 2 ** (quantspec['limpbitwidth'] - 1) + 1) if quantspec['signed'] else 0
            n_levels = 2 ** quantspec['limpbitwidth'] - 1

        else:
            raise ValueError(f"[QuantLib] Dictionary QuantSpec with keys {quantspec_keys} is not supported.")

    # string spec? ('binary' or 'ternary')
    elif isinstance(quantspec, str):

        if quantspec == 'binary':
            step = 2
            offset = -1
            n_levels = 2

        elif quantspec == 'ternary':
            step = 1
            offset = -1
            n_levels = 3

        else:
            raise ValueError(
                f"[QuantLib] String QuantSpecs should be either 'binary' or 'ternary', but {quantspec} was specified.")

    else:
        raise TypeError(f"[QuantLib] QuantSpec can be only a string, a tuple of integers or a dictionary, but {type(quantspec)} was specified.")

    return QRange(n_levels, offset, step)
