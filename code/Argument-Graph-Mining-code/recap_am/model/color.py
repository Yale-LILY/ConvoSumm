import colorsys
import itertools
import math
from fractions import Fraction
from typing import Generator

# https://stackoverflow.com/a/13781114


def _zenos_dichotomy():
    """
    http://en.wikipedia.org/wiki/1/2_%2B_1/4_%2B_1/8_%2B_1/16_%2B_%C2%B7_%C2%B7_%C2%B7
    """
    for k in itertools.count():
        yield Fraction(1, 2 ** k)


def _getfracs():
    """
    [Fraction(0, 1), Fraction(1, 2), Fraction(1, 4), Fraction(3, 4), Fraction(1, 8), Fraction(3, 8), Fraction(5, 8), Fraction(7, 8), Fraction(1, 16), Fraction(3, 16), ...]
    [0.0, 0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875, 0.0625, 0.1875, ...]
    """
    yield 0
    for k in _zenos_dichotomy():
        i = k.denominator  # [1,2,4,8,16,...]
        for j in range(1, i, 2):
            yield Fraction(j, i)


def _bias(x):
    return (math.sqrt(x / 3) / Fraction(2, 3) + Fraction(1, 3)) / Fraction(
        6, 5
    )  # can be used for the v in hsv to map linear values 0..1 to something that looks equidistant


def _genhsv(h):
    for s in [Fraction(6, 10)]:  # optionally use range
        for v in [Fraction(8, 10), Fraction(5, 10)]:  # could use range too
            yield (h, s, v)  # use bias for v here if you use range


def _genrgb(x):
    return colorsys.hsv_to_rgb(*x)


def _genhtml(x):
    uints = map(lambda y: int(y * 255), x)
    return "rgb({},{},{})".format(*uints)


def _clamp(x):
    return max(0, min(x, 255))


def _genhex(x):
    uints = tuple(map(lambda y: int(y * 255), x))

    return "#{0:02x}{1:02x}{2:02x}".format(
        _clamp(uints[0]), _clamp(uints[1]), _clamp(uints[2])
    )


_flatten = itertools.chain.from_iterable


def gethsv() -> Generator[str, None, None]:
    return _flatten(map(_genhsv, _getfracs()))


def getrgb() -> Generator[str, None, None]:
    return map(_genrgb, gethsv())


def gethtml() -> Generator[str, None, None]:
    return map(_genhtml, getrgb())


def gethex() -> Generator[str, None, None]:
    return map(_genhex, getrgb())


if __name__ == "__main__":
    print(list(itertools.islice(gethtml(), 100)))
