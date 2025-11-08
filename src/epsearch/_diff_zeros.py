import warnings
from collections.abc import Callable, Sequence
from typing import Any, Literal

import attrs
import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from ss_hankel import SSHKwargs, ss_h_circle

from ._branching import (
    BoundaryGenerator,
    Circle,
    FindExceptionalPointsRecursivelyResult,
    find_branching_points_recursively,
    is_analytic,
)
from ._cycle import Cycles, TNumber, get_cycles


def find_branching_points_using_zeros_ssh(
    f_eigvals: Callable[[NDArray[TNumber]], NDArray[TNumber]],
    /,
    **kwargs: SSHKwargs,
) -> Sequence[TNumber]:
    """
    Search for branching points using zero-finding methods.

    Parameters
    ----------
    f_eigvals : Callable[[NDArray[Any]], NDArray[Any]]
        A function that takes a batch of parameters and returns the eigenvalues.
    **kwargs : SSHKwargs
        Keyword arguments for the Sakurai-Sugiura method.

    Returns
    -------
    Sequence[TNumber]
        The branching points.

    """

    def f(p: NDArray[TNumber], /) -> NDArray[TNumber]:
        cycles: Cycles[TNumber] = get_cycles(f_eigvals(p))
        if cycles.max_cycle_length == 1:
            raise ValueError("No branching points found.")
        cycle_diffs = np.stack(
            [
                np.sum(cycle, axis=0) ** len(cycle) - np.prod(cycle * len(cycle), axis=0)
                for cycle in cycles.cycles
                if cycle.shape[0] > 1
            ],
            axis=-1,
        )
        return cycle_diffs[..., np.diag(np.arange(cycle_diffs.shape[-1]))]

    return ss_h_circle(f, **kwargs).eigval


def find_branching_points_using_zeros_scipy(
    f_eigvals: Callable[[NDArray[TNumber]], NDArray[TNumber]],
    x0: TNumber,
    eigval: TNumber,
    multiplicity: int,
    /,
    **kwargs: Any,
) -> TNumber:
    """
    Search for branching points using scipy's root-finding methods.

    Parameters
    ----------
    f_eigvals : Callable[[NDArray[Any]], NDArray[Any]]
        A function that takes a batch of parameters and returns the eigenvalues.
    x0 : TNumber
        The initial guess for the root-finding method.
    eigval : TNumber
        The expected eigenvalue at the branching point.
    multiplicity : int
        The multiplicity of the eigenvalue at the branching point.
    **kwargs : Any
        Keyword arguments for ``scipy.optimize.root``.

    Returns
    -------
    Sequence[TNumber]
        The branching points.

    """

    def f(p: NDArray[TNumber], /) -> NDArray[TNumber]:
        p = p[0] + 1j * p[1]
        eigvals = f_eigvals(p[None])[0]
        argsort = np.argsort(np.abs(eigvals - eigval))
        eigvals_near = eigvals[argsort[:multiplicity]]
        objective = np.abs(
            np.sum(eigvals_near, axis=0) ** multiplicity
            - np.prod(eigvals_near * multiplicity, axis=0)
        )
        return np.stack((objective.real, objective.imag), axis=0)

    res = scipy.optimize.root(f, np.stack((x0.real, x0.imag)), **kwargs).x
    return res[0] + 1j * res[1]


def find_branching_points_recursively_hybrid(
    f_eigvals: Callable[[Sequence[TNumber]], Sequence[Sequence[TNumber]]],
    f_boundary: BoundaryGenerator[Circle, TNumber],
    /,
    f_plot: Callable[[int | None, int | None], None] | None = None,
    rtol_analytic: float | None = None,
    ssh_kwargs: SSHKwargs | None = None,
    scipy_kwargs: Any | None = None,
    method: Literal["ssh", "scipy"] = "scipy",
    depth_first: bool = False,
    depth_first_and_break: bool = False,
) -> FindExceptionalPointsRecursivelyResult[Circle, TNumber]:
    """
    Search for branching points recursively.

    Parameters
    ----------
    f_eigvals : Callable[[Sequence[TNumber]], Sequence[Sequence[TNumber]]]
        A function that takes a batch of parameters and returns the eigenvalues.
    f_boundary : BoundaryGenerator[Circle]
        A function that takes the mapping key and whether a branching point is found
        inside the boundary, and returns the new boundaries
    f_plot : Callable[[int | None, int | None], None], optional
        A function that takes the iteration number and boundary key
        and plots the boundaries, by default None
    rtol_analytic : float, optional
        The relative tolerance for the analytic check,
        by default None.
    ssh_kwargs : SSHKwargs, optional
        Keyword arguments for the Sakurai-Sugiura method.
    scipy_kwargs : Any, optional
        Keyword arguments for the scipy root-finding method.
    method : Literal["ssh", "scipy"], optional
        The method to use for finding branching points,
        by default "scipy". If "ssh", the Sakurai-Sugiura method is used.
        If "scipy", scipy's root-finding method is used.
    depth_first : bool, optional
        Whether to use depth-first search for finding branching points,
        by default False. If True, the search will be depth-first,
        otherwise it will be breadth-first.
    depth_first_and_break : bool, optional
        Whether to use depth-first search and break
        when the first branching point is found,
        by default False. If True, the search will be depth-first
        and will break when the first branching point is found.

    Returns
    -------
    FindExceptionalPointsRecursivelyResult[Circle]
        The branching points.

    """

    def f_final(cycles: Cycles[TNumber]) -> bool:
        # check all cycle diffs ** cycle_length
        # are analytic
        for cycle in cycles.cycles:
            if cycle.shape[0] < 2:
                continue
            if cycle.shape[0] > 2:
                return False  # TODO: remove
            if not is_analytic(
                np.sum(cycle, axis=0) ** len(cycle) - np.prod(cycle * len(cycle), axis=0),
                rtol=rtol_analytic,
            ):
                return False
        return True

    result = find_branching_points_recursively(
        f_eigvals,
        f_boundary,
        f_final=f_final,
        f_plot=f_plot,
        eigvals_analytic=True,
        depth_first=depth_first,
        depth_first_and_break=depth_first_and_break,
    )
    branching_points: list[TNumber] = []
    for circle in result.keys:
        if method == "ssh":
            try:
                branching_points.extend(
                    find_branching_points_using_zeros_ssh(
                        f_eigvals,
                        circle_center=circle.center,
                        circle_radius=circle.radius,
                        **(ssh_kwargs or {}),
                    )
                )
            except ValueError as e:
                warnings.warn(f"No branching points found for {circle}.", source=e, stacklevel=2)
        elif method == "scipy":
            try:
                for connected_cycle in result.cycles[circle].connected_cycles:
                    if connected_cycle.shape[0] == 1:
                        continue
                    branching_points.append(
                        find_branching_points_using_zeros_scipy(
                            f_eigvals,
                            np.mean(result.boundaries[circle]),
                            np.mean(connected_cycle[0]),
                            connected_cycle.shape[0],
                            **(scipy_kwargs or {}),
                        )
                    )
            except ValueError as e:
                warnings.warn(f"No branching points found for {circle}.", source=e, stacklevel=2)
    return attrs.evolve(result, branching_points=np.sort_complex(branching_points))
