"""Figure 3: transverse bed-elevation profiles of the two competing bar modes.

Panel (a) is the central-bar eigenmode (``m = 2``) at ``k_A = 2.0``: symmetric
deposition centred on the channel axis with lateral scour.  Panel (b) is the
alternate-bar eigenmode (``m = 1``) at ``k_B = 1.0``: antisymmetric relief with a
nodal line on the channel axis.  Both are extracted from the depth-averaged
linear stability eigenproblem at the reference parameter set
(``beta = 15.0``, ``sigma = 0.15``, ``F = 0.5``, ``Cf = 0.01``, ``r_s = 0.5``) and
normalized so that ``max|zb_hat| = 1.0``, with ``zb_hat = 0`` the undisturbed
base level.

Run from the repository root::

    python code/fig03_eigenmode_structures.py
"""

import os
import sys

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit, chebval

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import linear_solver as swe
import diagnostics as diag
from figure_utils import (
    COLOR_ALTERNATE,
    COLOR_CENTRAL,
    save_publication_figure,
    set_publication_style,
)


def main():
    set_publication_style()

    params = {
        "beta": 15.0,
        "Cf": 0.01,
        "Fr": 0.5,
        "transport_exponent_u": 3.0,
        "slope_coeff": 0.5,
        "Exner_Coeff": 0.005,
    }
    sigma_val = 0.15
    N = 24
    D, y = diag.cheb_collocation(N)
    M = N + 1

    # Mode A: central bar, m = 2, k_A = 2.0.
    k_A = 2.0
    A_a, B_a = swe.assemble_swe_matrices(
        D, y, k_A, params["beta"], params["Cf"], params["Fr"],
        N_curv=0.0, sigma_width=sigma_val, params=params,
    )
    va, veca = la.eig(A_a, B_a)
    ma = np.isfinite(va) & (np.real(va) < 2.0)
    va, veca = va[ma], veca[:, ma]
    idx_a = np.argsort(va.real)[::-1]
    assert len(idx_a) >= 3, (
        f"Expected at least 3 physical modes for mode A, found {len(idx_a)}"
    )

    # Rank 2 in the spectrum is the symmetric central bar.
    vec_a = veca[:, idx_a[2]]
    zb_a = vec_a[3 * M:4 * M]

    # Phase rotation: crest purely real and positive on the channel axis.
    phi_a = np.angle(zb_a[N // 2])
    zb_a_rot = (zb_a * np.exp(-1j * phi_a)).real
    if zb_a_rot[N // 2] < 0:
        zb_a_rot *= -1

    # Mode B: alternate bar, m = 1, k_B = 1.0.
    k_B = 1.0
    A_b, B_b = swe.assemble_swe_matrices(
        D, y, k_B, params["beta"], params["Cf"], params["Fr"],
        N_curv=0.0, sigma_width=sigma_val, params=params,
    )
    vb, vecb = la.eig(A_b, B_b)
    mb = np.isfinite(vb) & (np.real(vb) < 2.0)
    vb, vecb = vb[mb], vecb[:, mb]
    idx_b = np.argsort(vb.real)[::-1]
    assert len(idx_b) >= 2, (
        f"Expected at least 2 physical modes for mode B, found {len(idx_b)}"
    )

    # Rank 1 in the spectrum is the antisymmetric alternate bar.
    vec_b = vecb[:, idx_b[1]]
    zb_b = vec_b[3 * M:4 * M]

    phi_b = np.angle(zb_b[0])
    zb_b_rot = (zb_b * np.exp(-1j * phi_b)).real

    # Spectral reconstruction through the interior nodes, with exact parity
    # imposed, removes the boundary collocation artifacts of the discrete
    # Neumann bank condition.
    y_int = y[1:-1]
    c_a = chebfit(y_int, zb_a_rot[1:-1], deg=4)
    c_a[1::2] = 0.0  # exact even symmetry

    c_b = chebfit(y_int, zb_b_rot[1:-1], deg=3)
    c_b[0::2] = 0.0  # exact odd symmetry

    y_fine = np.linspace(-1, 1, 201)
    za_fine = chebval(y_fine, c_a)
    zb_fine = chebval(y_fine, c_b)

    norm_a = np.max(np.abs(za_fine))
    za_fine = za_fine / norm_a if norm_a > 1e-12 else za_fine
    norm_b = np.max(np.abs(zb_fine))
    zb_fine = zb_fine / norm_b if norm_b > 1e-12 else zb_fine

    fig, axes = plt.subplots(1, 2, figsize=(6.5, 2.7), sharey=True)

    ax1 = axes[0]
    ax1.plot(y_fine, za_fine, "-", color=COLOR_CENTRAL, linewidth=1.8,
             label=r"Central bar ($m=2$)")
    ax1.fill_between(y_fine, 0, za_fine, where=(za_fine > 0), alpha=0.25,
                     color=COLOR_CENTRAL)
    ax1.fill_between(y_fine, 0, za_fine, where=(za_fine < 0), alpha=0.25,
                     color=COLOR_CENTRAL)
    ax1.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax1.set_xlabel(r"Normalized transverse coordinate $\zeta$")
    ax1.set_ylabel(r"Bed elevation $\hat{z}_b$")
    ax1.set_title(r"(a) Central bar ($k_A = 2.0$)", fontsize=10)
    ax1.set_xlim(-1, 1)
    ax1.set_ylim(-1.25, 1.25)
    ax1.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax1.text(0.0, 1.02, r"Symmetric ($m = 2$)", ha="center", fontsize=8.5,
             style="italic")

    ax2 = axes[1]
    ax2.plot(y_fine, zb_fine, "-", color=COLOR_ALTERNATE, linewidth=1.8,
             label=r"Alternate bar ($m=1$)")
    ax2.fill_between(y_fine, 0, zb_fine, where=(zb_fine > 0), alpha=0.25,
                     color=COLOR_ALTERNATE)
    ax2.fill_between(y_fine, 0, zb_fine, where=(zb_fine < 0), alpha=0.25,
                     color=COLOR_ALTERNATE)
    ax2.axhline(0, color="gray", linestyle=":", linewidth=0.8)
    ax2.set_xlabel(r"Normalized transverse coordinate $\zeta$")
    ax2.set_title(r"(b) Alternate bar ($k_B = 1.0$)", fontsize=10)
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1.25, 1.25)
    ax2.tick_params(labelleft=True)
    ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax2.text(-0.35, 1.02, r"Antisymmetric ($m = 1$)", ha="center", fontsize=8.5,
             style="italic")

    plt.tight_layout()
    save_publication_figure(fig, "Fig3_eigenmode_structures")
    print("Figure 3 written.")


if __name__ == "__main__":
    main()
