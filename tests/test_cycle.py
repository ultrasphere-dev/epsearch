import numpy as np
import pytest
from numpy.testing import assert_allclose
from numpy.typing import NDArray

from epsearch import get_cycles
from epsearch._cycle import argmatch_from_closest


def test_cycle():
    eigval = [
        [-5, 1, 2, 3],
        [
            3.1,
            1.1,
            2.1,
        ],
        [-1.2, 0.2, 1.2, 2.2, 3.2],
        [1.1, 2.1, 3.1],
    ]
    cycles_expected = np.asarray(
        [[1, 1.1, 1.2, 1.1], [2, 2.1, 2.2, 2.1], [3, 3.1, 3.2, 3.1]],
    )
    cycles = get_cycles(eigval)  # type: ignore
    assert cycles.eigvals.shape == (3, 4)
    assert_allclose(cycles.eigvals, cycles_expected)


def test_defective():
    eigval = [
        [1, 1],
        [1, 1, 1],
        [1, 1, 1, 1],
        [1, 1, 1],
        [1, 2],
    ]
    cycles_expected = np.asarray([[1, 1, 1, 1, 1], [1, 1, 1, 1, 2]])
    cycles = get_cycles(eigval)  # type: ignore
    assert cycles.eigvals.shape == (2, 5)
    assert_allclose(cycles.eigvals, cycles_expected)


@pytest.mark.parametrize(
    "x, y, expected",
    [
        (
            np.array([1, 2, 3]),
            np.array([2, 3, 1]),
            np.array([2, 0, 1]),
        ),
        (
            np.array([1, 2, 3]),
            np.array([1, 2, 3, 4]),
            np.array([0, 1, 2]),
        ),
    ],
)
def test_match_from_closest(
    x: NDArray[np.number], y: NDArray[np.number], expected: NDArray[np.number]
) -> None:
    result = argmatch_from_closest(x, y)
    assert_allclose(result, expected)
