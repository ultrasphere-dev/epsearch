from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray

from epsearch import (
    BoundaryGenerator,
    CirclesBoundary,
    RectsBoundary,
    find_branching_points_recursively,
)


@pytest.mark.parametrize(
    "boundary",
    [
        CirclesBoundary(center=0, radius=2, radius_min=0.1, n_points=128),
        RectsBoundary(center=0j, half_size=2 + 2j, half_size_min=0.1 + 0.1j, n_points_per_side=32),
    ],
)
def test_msd(
    boundary: BoundaryGenerator[Any, complex],
) -> None:
    path = Path(__file__).parent / ".cache"
    path.mkdir(exist_ok=True)
    A0 = np.asarray([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0], [1, -2, 0, 0]])
    A1 = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1]])

    def f(z: NDArray[Any]) -> NDArray[Any]:
        A = A0 + z[:, None, None] * A1
        return np.linalg.eigvals(A)

    def f_plot(i: int | None, k: int | None, /) -> None:
        plt.savefig(path / f"test_branching_msd_{i}_{k}_{boundary.__class__.__name__}.jpg")
        plt.clf()

    fig, _ = plt.subplots()
    res = find_branching_points_recursively(
        f,
        boundary,
        f_plot=f_plot,
        depth_first=True,
        depth_first_and_break=True,
    )
    res.plot()
    print(res.branching_points)
    print(f(np.asarray(res.branching_points)))
    fig.savefig(path / f"test_branching_msd_{boundary.__class__.__name__}.jpg")


@pytest.mark.parametrize(
    "boundary",
    [
        CirclesBoundary(center=3, radius=2, radius_min=0.1, n_points=128),
        RectsBoundary(
            center=3 + 0j, half_size=2 + 2j, half_size_min=0.1 + 0.1j, n_points_per_side=32
        ),
    ],
)
def test_random(
    boundary: BoundaryGenerator[Any, complex],
) -> None:
    N = 6
    n_params = 1
    rng = np.random.default_rng(1)
    matrices = rng.normal(size=(n_params + 1, N, N)) + 1j * rng.normal(size=(n_params + 1, N, N))

    def f(p: NDArray[Any]) -> NDArray[Any]:
        A = matrices[0] + p[:, None, None] * matrices[1]
        return np.linalg.eigvals(A)

    res = find_branching_points_recursively(f, boundary)
    res.plot()
    print(res.branching_points)
    print(f(np.asarray(res.branching_points)))
    path = Path(__file__).parent / ".cache"
    path.mkdir(exist_ok=True)
    plt.savefig(path / f"test_branching_random_{boundary.__class__.__name__}.jpg")
