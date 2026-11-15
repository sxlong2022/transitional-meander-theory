"""Figure 2: linear stability diagram across the width gradient ``sigma``.

For each ``sigma`` the discrete bar-wavenumber grid ``k`` is scanned and the
largest growth rate is retained separately for the central-bar branch
(``m = 2``, symmetric) and the alternate-bar branch (``m = 1``, antisymmetric).
Modes are classified from the symmetry of the bed-eigenfunction transverse
profile, so the two curves trace genuine branches rather than a single envelope.

Reference parameters: ``beta = 10.0``, ``F = 0.5``, ``Cf = 0.01``.

Run from the repository root::

    python code/fig02_phase_diagram.py
"""

import os
import sys

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import linear_solver as swe
import diagnostics as diag
from figure_utils import (
    COLOR_ALTERNATE,
    COLOR_CENTRAL,
    get_publication_colors,
    save_publication_figure,
    set_publication_style,
)


def rotate_to_max_real(zb):
    """Rotate a complex profile so its largest component is real and positive."""
    idx = np.argmax(np.abs(zb))
    phi = np.angle(zb[idx])
    return np.real(zb * np.exp(-1j * phi))


def is_central_bar(zb, N):
    """Return ``True`` when the profile is a symmetric central-bar eigenmode."""
    max_abs = np.max(np.abs(zb))
    if max_abs < 1e-10:
        return False
    zb_r = rotate_to_max_real(zb)
    zb_norm = zb_r / max_abs
    sym = diag.calculate_symmetry_index(zb_r, N)
    std_val = np.std(zb_norm)
    center_val = zb_norm[N // 2]
    bank_val = 0.5 * (zb_norm[0] + zb_norm[-1])
    is_opposite = (center_val * bank_val < 0)
    return (sym < 0.2) and (std_val > 0.2) and is_opposite


def is_alternate_bar(zb, N):
    """Return ``True`` when the profile is an antisymmetric alternate-bar eigenmode."""
    max_abs = np.max(np.abs(zb))
    if max_abs < 1e-10:
        return False
    zb_r = rotate_to_max_real(zb)
    zb_norm = zb_r / max_abs
    sym = diag.calculate_symmetry_index(zb_r, N)
    std_val = np.std(zb_norm)
    center_val = zb_norm[N // 2]
    return (sym > 0.8) and (std_val > 0.1) and (abs(center_val) < 0.2)


def main():
    set_publication_style()

    params = {
        "beta": 10.0,
        "Cf": 0.01,
        "Fr": 0.5,
        "transport_exponent_u": 3.0,
    }
    N = 16

    sigmas = np.linspace(-0.1, 0.2, 31)
    test_ks = [0.4, 0.5, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]

    D, y = diag.cheb_collocation(N)

    results = {"sigma": [], "central": [], "alternate": []}

    print("Scanning sigma for the linear stability diagram ...")

    for s_val in sigmas:
        best_central = -999.0
        best_alternate = -999.0

        for k_test in test_ks:
            A_mat, B_mat = swe.assemble_swe_matrices(
                D, y, k_test,
                params["beta"], params["Cf"], params["Fr"],
                N_curv=0.0, sigma_width=s_val, params=params,
            )

            vals, vecs = la.eig(A_mat, B_mat)

            # Discard the infinite algebraic eigenvalues of the bank constraints
            # plus non-physical wall modes with implausibly large growth rates.
            valid = np.isfinite(vals) & (np.real(vals) < 10.0)
            vals = vals[valid]
            vecs = vecs[:, valid]

            idx = np.argsort(np.real(vals))[::-1]
            M = N + 1

            for i in range(min(10, len(idx))):
                mode_id = idx[i]
                gr = np.real(vals[mode_id])
                vec = vecs[:, mode_id]
                zb = vec[3 * M:4 * M]

                if is_central_bar(zb, N) and gr > best_central:
                    best_central = gr
                if is_alternate_bar(zb, N) and gr > best_alternate:
                    best_alternate = gr

        if best_central < -100:
            best_central = np.nan
        if best_alternate < -100:
            best_alternate = np.nan

        results["sigma"].append(s_val)
        results["central"].append(best_central)
        results["alternate"].append(best_alternate)

    colors = get_publication_colors()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    ax.plot(results["sigma"], results["central"], marker="o", color=COLOR_CENTRAL,
            markersize=4, linewidth=1.2, label="Central bar (symmetric)")
    ax.plot(results["sigma"], results["alternate"], marker="s",
            color=COLOR_ALTERNATE, markersize=4, linewidth=1.2,
            label="Alternate bar (antisymmetric)")

    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle=":", linewidth=0.8)

    ax.set_xlabel(r"Width gradient $\sigma$")
    ax.set_ylabel(r"Growth rate $\sigma_g = \mathrm{Re}(\lambda)$")
    ax.set_xlim(-0.1, 0.2)
    ax.set_ylim(-0.5, 5.5)
    ax.legend(loc="upper right", fontsize=8, frameon=True)

    plt.tight_layout()
    save_publication_figure(fig, "Fig2_phase_diagram")
    print("Figure 2 written.")


if __name__ == "__main__":
    main()
