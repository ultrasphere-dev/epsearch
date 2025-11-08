from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from epsearch import (
    CirclesBoundary,
    find_branching_points_recursively,
)


def test_msd():
    path = Path(__file__).parent / ".cache"
    path.mkdir(exist_ok=True)
    A0 = np.asarray([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0], [1, -2, 0, 0]])
    A1 = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1]])

    def f(z: NDArray[Any]) -> NDArray[Any]:
        A = A0 + z[:, None, None] * A1
        return np.linalg.eigvals(A)

    def f_plot(i: int | None, k: int | None, /) -> None:
        plt.savefig(path / f"test_branching_msd_{i}_{k}.jpg")
        plt.clf()

    fig, _ = plt.subplots()
    res = find_branching_points_recursively(
        f,
        CirclesBoundary(center=1, radius=3, radius_min=0.01, n_points=128),
        f_plot=f_plot,
        depth_first=True,
        depth_first_and_break=True,
    )
    res.plot()
    print(res.branching_points)
    print(f(np.asarray(res.branching_points)))
    fig.savefig(path / "test_branching_msd.jpg")


def test_random():
    N = 6
    n_params = 1
    rng = np.random.default_rng(1)
    matrices = rng.normal(size=(n_params + 1, N, N)) + 1j * rng.normal(size=(n_params + 1, N, N))

    def f(p: NDArray[Any]) -> NDArray[Any]:
        A = matrices[0] + p[:, None, None] * matrices[1]
        return np.linalg.eigvals(A)

    res = find_branching_points_recursively(
        f,
        CirclesBoundary(center=3, radius=2, radius_min=0.01, n_points=128),
    )
    res.plot()
    print(res.branching_points)
    print(f(np.asarray(res.branching_points)))
    path = Path(__file__).parent / ".cache"
    path.mkdir(exist_ok=True)
    plt.savefig(path / "test_branching_random.jpg")
