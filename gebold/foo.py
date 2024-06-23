"""Foo module."""
import numpy as np


def foo_sum(a: int, b: int) -> int:
    return np.sum([a, b])


def foo_substract(a: int, b: int) -> int:
    return np.subtract(a, b)


def foo_prod(a: int, b: int) -> int:
    return np.prod([a, b])


def foo_div(a: int, b: int) -> float:
    return np.divide(a, b)


def foo_mean(a: int, b: int) -> float:
    return np.mean([a, b])


def foo_median(a: int, b: int) -> float:
    return np.median([a, b])


def foo_min(a: int, b: int) -> int:
    return np.min([a, b])


def foo_max(a: int, b: int) -> int:
    return np.max([a, b])


def foo_abs(a: int) -> int:
    return np.abs([a])


def foo_sqrt(a: int) -> float:
    return np.sqrt([a])
