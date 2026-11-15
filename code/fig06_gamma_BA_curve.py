"""Figure 6: cross-coupling coefficient ``Re(gamma_BA)`` across the width gradient.

The coefficient is evaluated on the co-evolving branch ``k_A = k_B = k_c = 2.0``
with continuous eigenmode tracking across ``sigma``.  Converging reaches
(``sigma < 0``) show conventional mode competition with ``Re(gamma_BA) > 0``,
straight channels (``sigma = 0``) are near-neutral up to discretization residual,
and diverging reaches (``sigma >= 0.005``) show systematic cross-enhancement with
``Re(gamma_BA) < 0``.

Tracking is anchored at ``sigma = 0.15`` and proceeds in both directions, so each
sampled point follows the same physical branch rather than re-selecting the most
unstable eigenvalue.

Run from the repository root::

    python code/fig06_gamma_BA_curve.py
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
from landau_calculator import LandauCoupledCalculator
import diagnostics as diag
from figure_utils import (
    COLOR_ALTERNATE,
    COLOR_CENTRAL,
    COLOR_MIXED,
    get_publication_colors,
    save_publication_figure,
    set_publication_style,
)


def track_modes_and_compute_curve():
    """Evaluate ``Re(gamma_BA)`` on ``sigma in [-0.10, 0.20]`` along a tracked branch.

    Returns:
        ``(sigmas, gamma_ba)``, sorted arrays of the sampled width gradients and
        the corresponding real part of the cross-coupling coefficient.
    """
    N = 16
    D, y = diag.cheb_collocation(N)
    k_c = 2.0
    params = {
        "beta": 15.0,
        "Cf": 0.01,
        "Fr": 0.5,
        "transport_exponent_u": 3.0,
        "slope_coeff": 0.5,
        "Exner_Coeff": 0.005,
    }

    # Anchor both branches at the reference state sigma = 0.15.
    print("Anchoring reference modes at sigma = 0.15 ...")
    A_ref, B_ref = swe.assemble_swe_matrices(
        D, y, k_c, params["beta"], params["Cf"], params["Fr"],
        N_curv=0.0, sigma_width=0.15, params=params,
    )
    v_ref, vec_ref = la.eig(A_ref, B_ref)
    mask = np.isfinite(v_ref) & (np.real(v_ref) < 2.0)
    v_ref = v_ref[mask]
    vec_ref = vec_ref[:, mask]

    idx_a = np.argmin(np.abs(v_ref - (0.3085 - 5.9616j)))
    idx_b = np.argmin(np.abs(v_ref - (0.2083 - 7.0364j)))
    assert np.abs(v_ref[idx_a] - (0.3085 - 5.9616j)) < 0.05, (
        f"Mode A anchor mismatch: {v_ref[idx_a]}"
    )
    assert np.abs(v_ref[idx_b] - (0.2083 - 7.0364j)) < 0.05, (
        f"Mode B anchor mismatch: {v_ref[idx_b]}"
    )
    mode_A_ref = vec_ref[:, idx_a]
    mode_B_ref = vec_ref[:, idx_b]

    results = {}

    def step(sig, cur_A, cur_B):
        """Advance both branches one sigma step and evaluate the coefficient."""
        A, B = swe.assemble_swe_matrices(
            D, y, k_c, params["beta"], params["Cf"], params["Fr"],
            N_curv=0.0, sigma_width=sig, params=params,
        )
        v, vec = la.eig(A, B)
        m = np.isfinite(v) & (np.real(v) < 2.0)
        v = v[m]
        vec = vec[:, m]

        idx = np.argmax(np.abs(vec.conj().T @ B @ cur_A))
        cur_mA, cur_omA = vec[:, idx], v[idx]

        idx = np.argmax(np.abs(vec.conj().T @ B @ cur_B))
        cur_mB, cur_omB = vec[:, idx], v[idx]

        calc = LandauCoupledCalculator(params, N, sigma_width=sig)
        g = calc.compute_full_coefficients(cur_mA, cur_omA, cur_mB, cur_omB,
                                          k_c, k_c)
        return np.real(g[2]), cur_mA, cur_mB

    # Backward sweep from sigma = 0.15 down to -0.10.
    cur_A, cur_B = mode_A_ref, mode_B_ref
    for sig in np.arange(0.15, -0.101, -0.005):
        value, cur_A, cur_B = step(float(sig), cur_A, cur_B)
        results[round(float(sig), 4)] = value

    # Forward sweep from sigma = 0.155 up to 0.20.
    cur_A, cur_B = mode_A_ref, mode_B_ref
    for sig in np.arange(0.155, 0.201, 0.005):
        value, cur_A, cur_B = step(float(sig), cur_A, cur_B)
        results[round(float(sig), 4)] = value

    sorted_sigmas = np.array(sorted(results.keys()))
    sorted_gamma = np.array([results[s] for s in sorted_sigmas])
    return sorted_sigmas, sorted_gamma


def main():
    set_publication_style()
    colors = get_publication_colors()

    sigmas, gamma_BA = track_modes_and_compute_curve()

    print("\nSampled reference values:")
    idx_005 = np.argmin(np.abs(sigmas - 0.05))
    idx_015 = np.argmin(np.abs(sigmas - 0.15))
    print(f"  sigma = {sigmas[idx_005]:.3f} -> Re(gamma_BA) = {gamma_BA[idx_005]:.6f}")
    print(f"  sigma = {sigmas[idx_015]:.3f} -> Re(gamma_BA) = {gamma_BA[idx_015]:.6f}")

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Cross-enhancement regime in diverging reaches.
    diverging = sigmas >= 0
    ax.fill_between(sigmas[diverging], gamma_BA[diverging], 0,
                    where=gamma_BA[diverging] < 0,
                    color=COLOR_ALTERNATE, alpha=0.25,
                    label=r"Cross-enhancement ($\mathrm{Re}(\gamma_{BA}) < 0$)")

    # Competition regime in converging reaches.
    converging = sigmas <= 0
    ax.fill_between(sigmas[converging], gamma_BA[converging], 0,
                    where=gamma_BA[converging] > 0,
                    color="lightgray", alpha=0.5,
                    label=r"Mode competition ($\mathrm{Re}(\gamma_{BA}) > 0$)")

    ax.plot(sigmas, gamma_BA, "-", color=COLOR_ALTERNATE, linewidth=1.8)
    ax.plot(sigmas[::2], gamma_BA[::2], "o", color=COLOR_ALTERNATE, markersize=3.5)

    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.axvline(0, color="k", linestyle=":", linewidth=0.8, alpha=0.5)

    ax.plot(0.05, gamma_BA[idx_005], "s", color=COLOR_CENTRAL, markersize=6,
            label=rf"$\sigma = 0.05$ ($\mathrm{{Re}}(\gamma_{{BA}}) = "
                  rf"{gamma_BA[idx_005]:.4f}$)")
    ax.plot(0.15, gamma_BA[idx_015], "^", color=COLOR_MIXED, markersize=6,
            label=rf"$\sigma = 0.15$ ($\mathrm{{Re}}(\gamma_{{BA}}) = "
                  rf"{gamma_BA[idx_015]:.4f}$)")

    ax.annotate(r"Converging: competition" + "\n" + r"$\mathrm{Re}(\gamma_{BA}) > 0$",
                xy=(-0.05, 0.08), fontsize=7.5, ha="center", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="gray", lw=0.5))

    ax.annotate(r"Diverging: cross-enhancement" + "\n" + r"$\mathrm{Re}(\gamma_{BA}) < 0$",
                xy=(0.08, -0.06), fontsize=7.5, ha="center", style="italic",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                          alpha=0.8, edgecolor="gray", lw=0.5))

    ax.set_xlabel(r"Width gradient $\sigma$")
    ax.set_ylabel(r"Cross-coupling coefficient $\mathrm{Re}(\gamma_{BA})$")
    ax.set_xlim(-0.10, 0.20)
    ax.set_ylim(-0.16, 0.18)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right", fontsize=7.5, frameon=True, framealpha=0.9)

    plt.tight_layout()
    save_publication_figure(fig, "Fig6_gamma_BA_curve")
    print("Figure 6 written.")


if __name__ == "__main__":
    main()
