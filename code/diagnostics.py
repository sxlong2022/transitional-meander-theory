"""Diagnostic metrics for classifying bar modes from a bed perturbation.

Two questions recur when scanning a linear stability spectrum:

1. Is the bed perturbation symmetric or antisymmetric about the channel axis?
   Symmetric modes are central bars (transverse mode number ``m = 2``);
   antisymmetric modes are alternate bars (``m = 1``).
2. Where does the perturbation peak in the transverse direction?

The functions here answer both questions from a single transverse profile
sampled on the Chebyshev collocation grid, ordered from ``zeta = +1`` to
``zeta = -1``.
"""

import numpy as np


def cheb_collocation(N):
    """Return the Chebyshev differentiation matrix and its collocation nodes.

    Args:
        N: Polynomial degree; the grid has ``N + 1`` nodes.

    Returns:
        ``(D, x)`` with ``D`` the ``(N + 1, N + 1)`` differentiation matrix and
        ``x`` the nodes ordered from ``+1`` to ``-1``.
    """
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack(([2], np.ones(N - 1), [2])) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x


def calculate_symmetry_index(zb, N):
    """Measure how far a bed profile departs from even symmetry.

    The profile is split at the channel axis and the left half is compared with
    the mirrored right half.  Perfect even symmetry gives ``0``; a pure odd
    profile gives a large value.

    Args:
        zb: Bed elevation profile on the collocation grid, ordered ``+1 -> -1``.
        N: Polynomial degree used to build the profile; ``len(zb) == N + 1``.

    Returns:
        Symmetry index, ``0`` for a purely symmetric (central) profile and
        increasing toward odd (alternate) profiles.
    """
    mid = N // 2
    val_left = zb[:mid]
    val_right_reversed = zb[-(mid):][::-1]

    # Even-ness: |L - R| approaches zero for a symmetric profile.
    diff_even = np.mean(np.abs(val_left - val_right_reversed))

    norm = np.mean(np.abs(val_left)) + 1e-10
    return diff_even / norm


def find_bar_axis(zb, y_grid):
    """Return the transverse coordinate of the largest bed perturbation.

    The phase of a linear eigenmode is arbitrary, so the peak is located on the
    amplitude envelope rather than on the real part alone.

    Args:
        zb: Bed elevation profile on the collocation grid.
        y_grid: Transverse coordinates matching ``zb``.

    Returns:
        Transverse coordinate ``zeta`` at which ``|zb|`` is largest.
    """
    return y_grid[np.argmax(np.abs(zb))]


def classify_mode(symmetry_index, growth_rate):
    """Classify a mode from its symmetry index and growth rate.

    Args:
        symmetry_index: Output of :func:`calculate_symmetry_index`.
        growth_rate: Real part of the eigenvalue associated with the mode.

    Returns:
        ``1`` for a central bar, ``2`` for an alternate bar, ``3`` for a
        boundary-dominated mode with an implausibly large growth rate.
    """
    if growth_rate > 0.5:
        return 3
    if symmetry_index < 0.2:
        return 1
    return 2
