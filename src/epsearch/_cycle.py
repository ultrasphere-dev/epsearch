from collections.abc import Sequence
from typing import Generic, TypeVar

import attrs
import networkx as nx
import numpy as np
from numpy.ma import MaskedArray

TNumber = TypeVar("TNumber", bound=np.number)


@attrs.frozen(kw_only=True)
class Cycles(Generic[TNumber]):
    """Cycles of the eigenvalues."""

    cycles: list[np.ndarray[tuple[int, int], np.dtype[TNumber]]]
    """Cycles of the eigenvalues of length num_cycles.

    Ordered by length in descending order of shape (cycle_length, num_points)."""
    graph: nx.DiGraph
    """Graph of the cycles. If eigvals[i, -1] ~ eigvals[j, 0], then
    there is an edge from i to j."""
    eigvals: np.ndarray[tuple[int, int], np.dtype[TNumber]]
    """Continuous eigenvalues of shape (num_complete_eigenvalues, num_points)."""
    incomplete_eigvals: MaskedArray[tuple[int, int], np.dtype[TNumber]]
    """Continuous eigenvalues of shape (num_eigenvalues, num_points)
    with some NaN values."""

    @property
    def connected_cycles(self) -> list[np.ndarray[tuple[int], np.dtype[TNumber]]]:
        """
        List of connected sequences of eigenvalues of length num_cycles.

        Prdered by length in descending order same as self.cycles
        of shape (cycle_length, num_points).
        """
        return [
            np.stack(
                [np.concatenate(np.roll(cycle, -i, axis=0)) for i in range(cycle.shape[0])],
                axis=0,
            )
            for cycle in self.cycles
        ]

    @property
    def cycle_lengths(self) -> list[int]:
        """Lengths of the cycles in descending order, same as self.cycles."""
        return [cycle.shape[0] for cycle in self.cycles]

    @property
    def max_cycle_length(self) -> int:
        """Maximum length of the cycles."""
        if len(self.cycle_lengths) == 0:
            return 0
        return np.max(self.cycle_lengths)


def argmatch_from_closest(
    x: np.ndarray[tuple[int], np.dtype[TNumber]],
    y: np.ndarray[tuple[int], np.dtype[TNumber]],
    /,
) -> MaskedArray[tuple[int], np.dtype[TNumber]]:
    """
    Match the elements of x and y from the closest without duplicates.

    Parameters
    ----------
    x : np.ndarray[tuple[int], np.dtype[TNumber]]
        First array of shape (n,).
    y : np.ndarray[tuple[int], np.dtype[TNumber]]
        Second array of shape (m,).

    Returns
    -------
    MaskedArray[tuple[int], np.dtype[TNumber]]
        Array of shape (n,).
        (x[i], y[result[i]]) is the pair.

    """
    if x.ndim != 1:
        raise ValueError(f"{x.ndim=} should be 1")
    if y.ndim != 1:
        raise ValueError(f"{y.ndim=} should be 1")
    dist = np.ma.masked_array(np.abs(x[:, None] - y[None, :]), mask=False)
    result = np.full_like(x, -1, dtype=int)
    for _ in range(min(len(x), len(y))):
        idx = np.unravel_index(dist.argmin(), dist.shape)
        result[idx[0]] = idx[1]
        dist.mask[idx[0], :] = np.inf
        dist.mask[:, idx[1]] = np.inf
    result = np.ma.masked_array(result, mask=result == -1)
    return result


def argmatch_from_closest_masked(
    x: MaskedArray[tuple[int], np.dtype[TNumber]],
    y: np.ndarray[tuple[int], np.dtype[TNumber]],
) -> MaskedArray[tuple[int], np.dtype[TNumber]]:
    """
    Match the elements of x and y from the closest without duplicates.

    Parameters
    ----------
    x : MaskedArray[tuple[int], np.dtype[TNumber]]
        The first array of shape (n,).
    y : np.ndarray[tuple[int], np.dtype[TNumber]],
        The second array. must be smaller than x.

    Returns
    -------
    MaskedArray[tuple[int], np.dtype[TNumber]]
        (x[i], y[result[i]]) is the pair.
        If x.mask.sum() < len(y), the indices of
        the remaining y elements are added to the extra indices.

    """
    if x.ndim != 1:
        raise ValueError(f"{x.ndim=} should be 1")
    if y.ndim != 1:
        raise ValueError(f"{y.ndim=} should be 1")
    if x.shape[0] < y.shape[0]:
        raise ValueError(f"{x.shape[0]=} < {y.shape[0]=}")

    arg = np.ma.masked_array(np.full_like(x, -1, dtype=int), mask=True)
    arg[~x.mask] = argmatch_from_closest(x[~x.mask], y)
    if y.shape[0] > x.mask.sum():
        arg_new = np.asarray(list(set(np.arange(len(y))) - set(arg[~arg.mask])))
        arg[x.mask.nonzero()[0][: len(arg_new)]] = np.asarray(arg_new)
    return arg


def get_cycles(
    eigvals: Sequence[Sequence[TNumber]] | np.ndarray[tuple[int, int], np.dtype[TNumber]],
    /,
) -> Cycles[TNumber]:
    """
    Get cycles from the eigenvalues for each point.

    Parameters
    ----------
    eigvals : Sequence[Sequence[TNumber]] | np.ndarray[tuple[int, int], np.dtype[TNumber]]
        A (ordered) sequence which contains the eigenvalues for each point.

    Returns
    -------
    bool
        Whether there is a branching inside the boundary.

    """
    # cycle the eigenvalues so that
    # forall i. len(eigenvalues[0]) <= len(eigenvalues[i])
    # eigv is a list which contains the eigenvalues for each point
    eig_list: list[np.ndarray[tuple[int], np.dtype[TNumber]]] = [
        np.asarray(eigs) for eigs in eigvals
    ]
    del eigvals
    eig_max_count = np.max([len(eigs) for eigs in eig_list])

    # reorder eigenvalues so that the distances between
    # eigenvalues_sorted[i] and eigenvalues_sorted[i+1] are minimized
    # [B, eigvals]
    eigvals_c_incomp = np.ma.masked_array(
        np.full(
            (len(eig_list), eig_max_count),
            np.nan,
            dtype=np.common_type(*eig_list),
            device=eig_list[0].device,
        ),
        mask=True,
    )
    for i, eigs in enumerate(eig_list):
        if i == 0:
            eigvals_c_incomp[i, : len(eigs)] = eigs
        else:
            arg = argmatch_from_closest_masked(eigvals_c_incomp[i - 1, :], eigs)
            eigvals_c_incomp[i, ~arg.mask] = eigs[arg[~arg.mask]]

    # for each continuous eigenvalues sequences, find a sequence
    # which start is the closest to the end
    eigvals_c = eigvals_c_incomp[:, ~eigvals_c_incomp.mask.any(axis=0)]
    arg = argmatch_from_closest(eigvals_c[-1, :], eigvals_c[0, :])
    G = nx.DiGraph(list(enumerate(arg[~arg.mask])))
    cycles = list(nx.simple_cycles(G))
    # order by length
    cycles = sorted(cycles, key=len, reverse=True)
    return Cycles(
        cycles=[eigvals_c[:, cycle].swapaxes(0, 1) for cycle in cycles],
        graph=G,
        eigvals=eigvals_c.T,
        incomplete_eigvals=eigvals_c_incomp.T,
    )
