from functools import partial
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pytest
from joblib import Parallel, delayed
from numpy.typing import NDArray
from ss_hankel import SSHKwargs, ss_h_circle
from tqdm_joblib import tqdm_joblib

from epsearch._branching import CirclesBoundary, count_duplicate
from epsearch._diff_zeros import (
    find_branching_points_recursively_hybrid,
    find_branching_points_using_zeros_ssh,
)


@pytest.mark.parametrize(
    "hybrid,depth_first,depth_first_and_break",
    [
        (True, True, True),
        (True, True, False),
        (True, False, False),
        (False, False, False),
    ],
)
@pytest.mark.parametrize("method", ["ssh", "scipy"])
def test_msd(
    hybrid: bool,
    method: Literal["ssh", "scipy"],
    depth_first: bool,
    depth_first_and_break: bool,
) -> None:
    A0 = np.asarray([[0, 0, 1, 0], [0, 0, 0, 1], [-1, 1, 0, 0], [1, -2, 0, 0]])
    A1 = np.asarray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, -1]])

    def f(p: NDArray[Any]) -> NDArray[Any]:
        A = A0 + p[:, None, None] * A1
        return np.linalg.eigvals(A)

    if hybrid:
        res = find_branching_points_recursively_hybrid(
            f,
            CirclesBoundary(center=0, radius=4, radius_min=0.001, n_points=128),
            ssh_kwargs=SSHKwargs(circle_n_points=128),
            method=method,
            depth_first=depth_first,
            depth_first_and_break=depth_first_and_break,
        )
        print(res.branching_points)
        print(f(np.asarray(res.branching_points)))
        res.plot()
        path = Path(__file__).parent / ".cache"
        path.mkdir(exist_ok=True)
        plt.savefig(
            path / f"test_diff_msd_{hybrid}_{method}_{depth_first}_{depth_first_and_break}.jpg"
        )
    else:
        if method == "scipy":
            pytest.skip("scipy not implemented yet")
        else:
            res = find_branching_points_using_zeros_ssh(  # type: ignore
                f,
                circle_n_points=128,
                circle_center=2 + 0.1j,
                circle_radius=0.6,
                max_order=5,
            )
        print(res, f(res))


@pytest.mark.skip(reason="Too heavy")
def test_random():
    path = Path(__file__).parent / ".cache"
    path.mkdir(exist_ok=True)
    N = 4
    n_order = 12
    rng = np.random.default_rng(1)
    coefs = rng.normal(size=(n_order, n_order, N, N)) + 1j * rng.normal(
        size=(n_order, n_order, N, N)
    )

    def f_plot(i: int | None, k: int | None, /) -> None:
        plt.savefig(path / f"test_diff_zeros_random_{i}_{k}.jpg")
        plt.clf()

    def f(p: NDArray[Any]) -> NDArray[Any]:
        def finner(x: NDArray[Any]) -> NDArray[Any]:
            x_ = (x.squeeze()[:, None] ** np.arange(n_order)[None, :])[:, None, :, None, None, None]
            p_ = (p[:, None] ** np.arange(n_order)[None, :])[None, :, None, :, None, None]
            A = (coefs[None, None, ...] * p_ * x_).sum(axis=2).sum(axis=2)
            return A

        eig = ss_h_circle(
            finner,
            circle_n_points=4096,
            circle_center=np.asarray([0]),
            circle_radius=np.asarray([3]),
            num_vectors=2,
            max_order=8,
        )
        return eig.eigval

    res = find_branching_points_recursively_hybrid(
        f,
        CirclesBoundary(center=0, radius=0.4, radius_min=0.001, n_points=512),
        ssh_kwargs=SSHKwargs(circle_n_points=128, max_order=4),
        f_plot=f_plot,
    )
    print(res.branching_points)
    if len(res.branching_points) > 0:
        print(f(np.asarray(res.branching_points)))
        print(count_duplicate(f(np.asarray(res.branching_points))))
    res.plot()
    plt.savefig(path / "test_diff_random.jpg")


@pytest.mark.skip(reason="Too heavy")
def test_random_linear():
    path = Path(__file__).parent / ".cache"
    path.mkdir(exist_ok=True)
    N = 4
    rng = np.random.default_rng(1)
    As = rng.normal(size=(3, N, N)) + 1j * rng.normal(size=(3, N, N))

    def f(p: NDArray[Any], q: NDArray[Any]) -> NDArray[Any]:
        p = np.asarray(p)[:, None, None]
        return np.linalg.eigvals(As[0, ...] + p * As[1, ...] + q * As[2, ...])

    def f2_plot(i: int | None, k: int | None, /) -> None:
        plt.savefig(path / f"test_diff_zeros_random_linear_{i}_{k}.jpg")
        plt.clf()

    def f2(p: NDArray[Any]) -> NDArray[Any]:
        """Return ep."""
        with tqdm_joblib(total=len(p)):
            return [
                x.branching_points
                for x in Parallel(n_jobs=-1)(
                    delayed(find_branching_points_recursively_hybrid)(
                        partial(f, q=p_),
                        CirclesBoundary(center=0, radius=10, radius_min=0.001, n_points=128),
                        ssh_kwargs=SSHKwargs(circle_n_points=128, max_order=2),
                    )
                    for p_ in p
                )
            ]

    res = find_branching_points_recursively_hybrid(
        f2,
        CirclesBoundary(center=0, radius=1, radius_min=0.001, n_points=128),
        ssh_kwargs=SSHKwargs(circle_n_points=128, max_order=2),
        f_plot=f2_plot,
    )
    res.plot()
