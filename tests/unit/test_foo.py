"""Unit tests for the foo module."""
import pytest

from gebold import foo


def test_foo_sum():
    assert foo.foo_sum(2, 2) == 4


def test_foo_substract():
    assert foo.foo_substract(5, 3) == 2


def test_foo_prod():
    assert foo.foo_prod(2, 3) == 6


def test_foo_div():
    assert foo.foo_div(10, 2) == 5.0


def test_foo_mean():
    assert foo.foo_mean(2, 4) == 3.0


def test_foo_median():
    assert foo.foo_median(2, 3) == 2.5


def test_foo_min():
    assert foo.foo_min(1, 2) == 1


def test_foo_max():
    assert foo.foo_max(1, 2) == 2


@pytest.mark.parametrize("a, expected", [(1, 1), (-1, 1)])
def test_foo_abs(a, expected):
    assert foo.foo_abs(a) == expected


@pytest.mark.parametrize("a, expected", [(4, 2.0), (9, 3.0)])
def test_foo_sqrt(a, expected):
    assert foo.foo_sqrt(a) == expected
