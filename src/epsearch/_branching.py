from collections.abc import Callable, Mapping, Sequence
from typing import Any, Generic, Protocol, TypeVar

import attrs
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from colormap_complex import colormap
from matplotlib_multicolored_line import colored_line
from numpy.typing import NDArray

from epsearch._cycle import Cycles, TNumber, get_cycles

TKey = TypeVar("TKey")
TNumber_co_ = TypeVar("TNumber_co_", bound=np.number, covariant=True)


def count_duplicate(a: NDArray[Any], /, *, eps: float = 1e-3) -> NDArray[Any]:
    """Count the number of duplicates."""
    a = np.abs(a[..., :, None] - a[..., None, :]) + np.eye(a.shape[-1]) * 1e10
    return (a.min(axis=-1) < eps).sum(axis=-1)


def contour_integral(values: NDArray[TNumber], /) -> NDArray[TNumber]:
    """
    Calculate contour integral.

    Parameters
    ----------
    values: NDArray[TNumber]
        Values of shape (..., num_points).

    Returns
    -------
    NDArray[TNumber]
        The contour integral of shape (...).

    """
    j = np.arange(values.shape[-1]) / values.shape[-1]
    weights = np.exp(2j * np.pi * j)
    cauchy = np.mean(values * weights, axis=-1)
    return cauchy


def is_analytic(
    values: NDArray[TNumber],
    *,
    rtol: float | None = None,
    integral: NDArray[TNumber] | None = None,
) -> bool:
    """
    Check if Cauchy's integral formula is fulfilled.

    Parameters
    ----------
    values : NDArray[TNumber]
        Values of shape (..., num_points).
    rtol : float, optional
        The relative tolerance, by default 1e-3.
    integral : NDArray[TNumber] | None, optional
        The contour integral of shape (...).

    Returns
    -------
    bool
        Whether Cauchy's integral formula is fulfilled
        for all sequences.

    """
    rtol = 1e-3 if rtol is None else rtol
    integral = contour_integral(values) if integral is None else integral
    return bool(np.all(np.abs(integral) < rtol * np.max(np.abs(values), axis=-1)))


class BoundaryGenerator(Protocol[TKey, TNumber_co_]):
    """A protocol for boundary generators."""

    def __call__(
        self, go_further: Mapping[TKey, bool], /
    ) -> tuple[Mapping[TKey, Sequence[TNumber_co_]], Sequence[TKey]]:
        """
        Divide-and-conquer search.

        Parameters
        ----------
        go_further : Mapping[TKey, bool]
            Whether to go further for each boundary.

        Returns
        -------
        tuple[Mapping[TKey, Sequence[TNumber_co_]], Sequence[TKey]]
            The new boundaries and the final keys.

        """
        ...

    def keys_to_values(self, keys: Sequence[TKey], /) -> Sequence[TNumber_co_]:
        """
        Get the final candidates.

        Parameters
        ----------
        keys : Sequence[TKey]
            The keys of the final candidates.

        Returns
        -------
        Sequence[TNumber_co_]
            The final candidates.

        """
        ...


@attrs.frozen(kw_only=True)
class FindExceptionalPointsRecursivelyResult(Generic[TKey, TNumber]):
    """The result of the recursive search."""

    boundaries: Mapping[TKey, Sequence[TNumber]]
    """The boundaries returned by the boundary generator."""
    eigvals: Mapping[TKey, Sequence[Sequence[TNumber]]]
    """The eigenvalues for each boundary."""
    cycles: Mapping[TKey, Cycles[TNumber]]
    """The cycles for each boundary."""
    generations: Mapping[TKey, int]
    """The generation of each boundary."""
    keys: Sequence[TKey]
    """The keys of the final candidates."""
    branching_points: Sequence[TNumber]
    """The branching points found."""

    def plot(
        self,
        text_contour_integral: bool = True,
        text_additional: Callable[[Cycles[TNumber]], str] | None = None,
        set_limits: bool = True,
    ) -> None:
        """Plot the boundaries and the eigenvalues."""
        plot(
            boundaries=self.boundaries,
            cycles=self.cycles,
            generations=self.generations,
            branching_points=self.branching_points,
            text_contour_integral=text_contour_integral,
            text_additional=text_additional,
            set_limits=set_limits,
        )


def plot(
    *,
    boundaries: Mapping[TKey, Sequence[TNumber]] | Sequence[TNumber],
    cycles: Mapping[TKey, Cycles[TNumber]] | Cycles[TNumber],
    generations: Mapping[TKey, int] | None = None,
    branching_points: Sequence[TNumber] | None = None,
    text_contour_integral: bool = True,
    text_additional: Callable[[Cycles[TNumber]], str] | None = None,
    set_limits: bool = False,
) -> None:
    """Plot the boundaries and the eigenvalues."""
    sns.set_theme()
    _, ax = plt.subplots(2, 2, figsize=(20, 20), layout="constrained")
    ax: Sequence[plt.Axes] = ax.reshape(-1)  # type: ignore
    if isinstance(boundaries, Mapping):
        boundaries_: Mapping[Any, Sequence[TNumber]] = boundaries
    else:
        boundaries_ = {None: boundaries}
    if isinstance(cycles, Cycles):
        cycles_: Mapping[Any, Cycles[TNumber]] = {None: cycles}
    else:
        cycles_ = cycles
    del boundaries, cycles
    cmap = colormap(type="oklch")
    has_multiple_generations = (
        generations is not None and np.unique(list(generations.values())).size > 1
    )
    has_multiple = len(boundaries_) > 1
    for ik, k in enumerate(boundaries_):
        i = generations.get(k, 0) if generations is not None else 0
        boundary_ = np.asarray(boundaries_[k])
        cycle_: Cycles[TNumber] = cycles_[k]
        if not has_multiple_generations:
            color = plt.get_cmap("twilight")(np.linspace(0, 1, len(boundary_)))[:, :3]
        else:
            color = cmap(
                np.linspace(0, 1, len(boundary_)),
                1 - i / (len(boundaries_) - 1),
            )

        ax[0].scatter(
            boundary_.real,
            boundary_.imag,
            c=color,
        )
        prefix = ""
        if not has_multiple:
            prefix = f"B{ik}-{prefix}"
        if not has_multiple_generations:
            prefix = f"G{i}-{prefix}"
        if cycle_.max_cycle_length > 1:
            ax[0].text(
                boundary_[0].real,
                boundary_[0].imag,
                f"{prefix}{cycle_.max_cycle_length}",
                fontsize=8,
            )
        ax[1].scatter(
            cycle_.incomplete_eigvals.real.flatten(),
            cycle_.incomplete_eigvals.imag.flatten(),
            marker="o",
            c=np.broadcast_to(color[None, :, :], (*cycle_.incomplete_eigvals.shape[:2], 3)).reshape(
                -1, 3
            ),
        )
        for cycle in cycle_.cycles:
            colored_line(
                cycle.real.T,
                cycle.imag.T,
                c=color[:, None, :],
                ax=ax[3 if cycle.shape[0] == 1 else 2],
            )
            for j in range(cycle.shape[0]):
                text = f"{prefix}C{cycle.shape[0]}-{j}"
                if cycle.shape[0] == 1:
                    continue
                if text_contour_integral and j == 0:
                    contour_integral_abs = np.abs(
                        contour_integral(
                            np.sum(cycle, axis=0) ** len(cycle)
                            - np.prod(cycle * len(cycle), axis=0)
                        )
                    )
                    text += f"\n∫: {contour_integral_abs:.3g}"
                ax[2].text(
                    cycle[j, 0].real,
                    cycle[j, 0].imag,
                    text,
                    fontsize=8,
                )
    if branching_points is not None:
        ax[0].plot(np.real(branching_points), np.imag(branching_points), "x")
    ax[0].set_title("Trace of the parameter (p)")
    ax[0].set_xlabel("Re p")
    ax[0].set_ylabel("Im p")
    ax[1].set_title("Scatter plot of the eigenvalues (λ)")
    ax[2].set_title("Trace of the eigenvalues \nwhich period > 1 (λ)")
    ax[3].set_title("Trace of the eigenvalues \nwhich period = 1 (λ)")
    for ax_ in ax[1:]:
        ax_.set_xlabel("Re λ")
        ax_.set_ylabel("Im λ")
    if set_limits:
        xlim = ax[1].get_xlim()
        ylim = ax[1].get_ylim()
        for ax_ in ax[2:]:
            ax_.set_xlim(xlim)
            ax_.set_ylim(ylim)


def find_branching_points_recursively(
    f_eigvals: Callable[[Sequence[TNumber]], Sequence[Sequence[TNumber]]],
    f_boundary: BoundaryGenerator[TKey, TNumber],
    /,
    *,
    f_go_further: Callable[[Cycles[TNumber]], bool] | None = None,
    f_final: Callable[[Cycles[TNumber]], bool] | None = None,
    f_plot: Callable[[int | None, int | None], None] | None = None,
    eigvals_analytic: bool = True,
    rtol_analytic: float | None = None,
    depth_first: bool = False,
    depth_first_and_break: bool = False,
) -> FindExceptionalPointsRecursivelyResult[TKey, TNumber]:
    """
    Search for branching points recursively.

    Parameters
    ----------
    f_eigvals : Callable[[Sequence[TNumber]], Sequence[Sequence[TNumber]]]
        A function that takes a batch of parameters and returns the eigenvalues.
    f_boundary : BoundaryGenerator[TKey]
        A function that takes the mapping key and whether a branching point is found
        inside the boundary, and returns the new boundaries
    f_go_further : Callable[[Cycles[TNumber]], bool], optional
        A function that takes the cycles and returns whether to go further,
        by default None
    f_final : Callable[[Cycles[TNumber]], bool], optional
        A function that takes the cycles and returns whether the boundary is final,
        by default None
    f_plot : Callable[[int | None, int | None], None], optional
        A function that takes the iteration number and boundary key
        and plots the boundaries, by default None
    eigvals_analytic : bool, optional
        Whether the eigenvalues are supposed to be analytic
        on the region except for the branching points,
        by default True.
        If True, the function will check if the eigenvalues
        follow Cauchy's integral formula as well.
    rtol_analytic : float, optional
        The relative tolerance for the analytic check,
        by default None.
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
    FindExceptionalPointsRecursivelyResult[TKey]
        The branching points.

    """
    if f_go_further is None:

        def f_go_further(cycles: Cycles[TNumber]) -> bool:
            return cycles.max_cycle_length > 1

    boundaries = dict(f_boundary({})[0])
    boundaries_stack: list[tuple[TKey, Sequence[TNumber]]] = list(boundaries.items())
    eigvals = {}
    cycles = {}
    generations = dict.fromkeys(boundaries.keys(), 0)
    final_keys: list[TKey] = []
    while boundaries_stack:
        # check has_branching for new boundaries
        k, boundary = boundaries_stack.pop()
        eigval = f_eigvals(boundary)
        if len(eigval) != len(boundary):
            raise ValueError(
                "f_eigvals must return the same number of eigenvalues "
                "as the number of points, but "
                f"len(f(points))={len(eigval)} != len(points)={len(boundary)}"
            )
        cycle: Cycles[TNumber] = get_cycles(eigval)
        cycles[k] = cycle
        eigvals[k] = eigval
        generation = generations[k]
        has_inside = f_go_further(cycle)
        final = has_inside and f_final(cycle) if f_final is not None else False

        # analytic check if eigvals_analytic is True and Circle boundary
        if isinstance(f_boundary, CirclesBoundary) and eigvals_analytic:
            has_inside = has_inside or not is_analytic(cycle.eigvals, rtol=rtol_analytic)

        # plot
        if f_plot is not None:
            plot(
                boundaries=boundary,
                cycles=cycle,
                generations=None,
            )
            f_plot(generation, len(cycles))

        if final:
            final_keys.append(k)
            boundaries_new: Mapping[TKey, Sequence[TNumber]] = {}
        else:
            # get new boundaries based on the previous result
            boundaries_new, final_keys_based_on_f_boundary = f_boundary({k: has_inside})
            final_keys.extend(final_keys_based_on_f_boundary)

        # break if depth_first and depth_first_and_break
        if depth_first_and_break and final_keys:
            break

        # append new boundaries and eigenvalues
        boundaries.update(boundaries_new)
        if not depth_first:
            boundaries_stack = list(boundaries_new.items()) + boundaries_stack
        else:
            boundaries_stack.extend(boundaries_new.items())
        generations.update(dict.fromkeys(boundaries_new.keys(), generation + 1))

    result = FindExceptionalPointsRecursivelyResult(
        boundaries={k: boundary for k, boundary in boundaries.items() if k in eigvals},
        eigvals=eigvals,
        cycles=cycles,
        generations=generations,
        keys=list(dict.fromkeys(final_keys)),  # remove duplicates
        branching_points=list(f_boundary.keys_to_values(final_keys)),
    )
    if f_plot is not None:
        result.plot()
        f_plot(None, None)
    return result


@attrs.frozen(kw_only=True)
class Circle:
    """A circle."""

    radius: float
    """The radius of the circle."""
    center: complex
    """The center of the circle."""


@attrs.frozen(kw_only=True)
class CirclesBoundary(BoundaryGenerator[Circle, complex]):
    """
    Divide-and-conquer search using circles.

    The circles circumscribe the rectangular region.
    The search region is a square
    [Re center - radius/sqrt(2), Re center + radius/sqrt(2)]
    x [Im center - radius/sqrt(2), Im center + radius/sqrt(2)].

    Parameters
    ----------
    center : complex
        The center of the circle.
    radius : float
        The radius of the circle.
    radius_min : float
        The radius threshold to stop the recursion.
    n_points : int
        The number of points on the circle.
    extra_ratio : float, optional
        The extra ratio to enlarge the circle
        to avoid the corners of the square to
        be missed, by default 0.1.
        Must be positive or zero.

    """

    center: complex
    radius: float
    radius_min: float
    n_points: int
    extra_ratio: float = 0.1

    def _circle(self, *, center: complex, radius: float) -> tuple[Circle, Sequence[complex]]:
        points = center + radius * (1 + self.extra_ratio) * np.exp(
            2j * np.pi * np.arange(self.n_points) / self.n_points
        )
        return Circle(center=center, radius=radius), points

    def __call__(
        self, go_further: Mapping[Circle, bool], /
    ) -> tuple[Mapping[Circle, Sequence[complex]], Sequence[Circle]]:
        """
        Divide-and-conquer search using circles.

        The circles circumscribe the rectangular region.
        """
        final_keys = []
        if not go_further:
            return dict([self._circle(center=self.center, radius=self.radius)]), []
        else:
            result: dict[Circle, Sequence[complex]] = {}
            for circle, branching in go_further.items():
                if not branching:
                    continue
                if circle.radius < self.radius_min:
                    final_keys.append(circle)
                    continue
                result.update(
                    dict(
                        [
                            self._circle(
                                center=circle.center
                                + circle.radius / 2 / np.sqrt(2) * (i + j * 1j),
                                radius=circle.radius / 2,
                            )
                            for i in [-1, 1]
                            for j in [-1, 1]
                        ]
                    )
                )
            return result, final_keys

    def keys_to_values(self, keys: Sequence[Circle], /) -> Sequence[complex]:
        """
        Get the final candidates.

        Parameters
        ----------
        keys : Sequence[Circle]
            The keys of the final candidates.

        Returns
        -------
        Sequence[complex]
            The final candidates.

        """
        centers = np.asarray([circle.center for circle in keys])
        radii = np.asarray([circle.radius for circle in keys])
        result = []
        for i in range(len(centers)):
            if (np.abs(centers[i] - centers[:i]) > radii[i] + radii[:i]).all():
                result.append(centers[i])
        return result
