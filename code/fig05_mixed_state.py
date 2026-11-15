"""Figure 5: reconstructed two-dimensional bed topography of the mixed state.

The reconstructed bed superposes the two physical linear eigenmodes,

.. math::

    Z_{\\mathrm{rec}}(s, \\zeta) = |A| \\hat{z}_A(\\zeta) \\cos(k_A s)
                                 + |B| \\hat{z}_B(\\zeta) \\cos(k_B s),

with the normal-form equilibrium amplitudes ``|A| = 20.42`` and ``|B| = 10.06``
from the regularized amplitude equations.  Mode A is the symmetric central bar at
``k_A = 2.0``; mode B is the antisymmetric alternate bar at ``k_B = 1.0``.  Positive
elevations denote bar deposition and negative values denote scour.  The relief is
normalized by ``max|Z_rec|``.

Run from the repository root::

    python code/fig05_mixed_state.py
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
import figure_utils as fu


def main():
    fu.set_publication_style()

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

    # 1. Physical mode A: central bar, m = 2, k_A = 2.0.
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

    vec_a = veca[:, idx_a[2]]
    zb_a = vec_a[3 * M:4 * M]
    phi_a = np.angle(zb_a[N // 2])
    zb_a_rot = (zb_a * np.exp(-1j * phi_a)).real
    if zb_a_rot[N // 2] < 0:
        zb_a_rot *= -1

    # 2. Physical mode B: alternate bar, m = 1, k_B = 1.0.
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

    vec_b = vecb[:, idx_b[1]]
    zb_b = vec_b[3 * M:4 * M]
    phi_b = np.angle(zb_b[0])
    zb_b_rot = (zb_b * np.exp(-1j * phi_b)).real
    if zb_b_rot[0] < 0:
        zb_b_rot *= -1

    # 3. Spectral reconstruction with exact parity.
    y_int = y[1:-1]
    c_a = chebfit(y_int, zb_a_rot[1:-1], deg=4)
    c_a[1::2] = 0.0
    c_b = chebfit(y_int, zb_b_rot[1:-1], deg=3)
    c_b[0::2] = 0.0

    Ny = 101
    y_fine = np.linspace(-1, 1, Ny)
    za_fine = chebval(y_fine, c_a)
    zb_fine = chebval(y_fine, c_b)
    max_a = np.max(np.abs(za_fine))
    max_b = np.max(np.abs(zb_fine))
    if max_a > 1e-12:
        za_fine /= max_a
    if max_b > 1e-12:
        zb_fine /= max_b

    # Regularized normal-form equilibrium amplitudes.
    amp_A = 20.42
    amp_B = 10.06

    # Three alternate-bar wavelengths span the plotted streamwise window.
    L_plot = 6.0 * np.pi
    Nx = 400
    x = np.linspace(0, L_plot, Nx)
    X, Y = np.meshgrid(x, y_fine)

    Z_A = amp_A * za_fine[:, None] * np.cos(k_A * X)
    Z_B = amp_B * zb_fine[:, None] * np.cos(k_B * X)
    Z_total = Z_A + Z_B
    Z_norm = Z_total / np.max(np.abs(Z_total))

    fig, ax = plt.subplots(figsize=(6.5, 2.2))
    levels = np.linspace(-1.0, 1.0, 31)
    im = ax.contourf(X, Y, Z_norm, levels=levels, cmap="RdBu_r", extend="both")

    ax.set_xlabel(r"Streamwise coordinate $s$")
    ax.set_ylabel(r"Transverse coordinate $\zeta$")
    ax.set_yticks([-1.0, 0.0, 1.0])
    ax.set_xlim(0, L_plot)
    ax.set_ylim(-1.0, 1.0)

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label(
        r"Normalized bed relief $Z_{\mathrm{rec}} / \max|Z_{\mathrm{rec}}|$",
        fontsize=8.5,
    )
    cbar.set_ticks([-1.0, -0.5, 0.0, 0.5, 1.0])

    plt.tight_layout()
    fu.save_publication_figure(fig, "Fig5_mixed_state")
    print("Figure 5 (mixed-state topography) written.")


if __name__ == "__main__":
    main()
