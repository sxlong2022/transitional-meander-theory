"""Recompute the Landau coefficient table across aspect ratio and width gradient.

For each ``(beta, sigma)`` pair the script identifies the central-bar and
alternate-bar eigenmodes, then evaluates the four coupled Landau coefficients and
writes one row per pair to ``data/landau_sweep_results.csv``.

Wavenumber pairing follows the natural subharmonic configuration: mode A at
``k_A = 2.0`` and mode B at ``k_B = 1.0``.  Coefficients follow the illustrative
single-carrier baseline assembly of main-text Table 2, passing only ``k_B = 1.0``
(``k_b=None``) so both operators are assembled on the ``k = 1.0`` carrier grid
and the second harmonic sits at ``2 k_B = 2.0`` (while ``mode_A`` and ``omega_A``
originate from the ``k_A = 2.0`` eigenproblem).
Run from the repository root::

    python code/recalculate_all.py
"""

import csv
import os
import sys

import numpy as np
import scipy.linalg as la

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_CODE_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")

if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import linear_solver as swe
import diagnostics as diag
from landau_calculator import LandauCoupledCalculator


def identify_mode(vecs, vals, idx_order, N, symmetric, sym_threshold=0.5):
    """Return the first eigenpair whose bed profile matches the requested parity.

    Args:
        vecs: Eigenvector matrix.
        vals: Eigenvalue vector.
        idx_order: Candidate indices, most unstable first.
        N: Chebyshev polynomial degree.
        symmetric: ``True`` to select an even profile, ``False`` for odd.
        sym_threshold: Symmetry-index threshold separating the two parities.

    Returns:
        ``(vector, eigenvalue)``, or ``(None, None)`` when no candidate matches.
    """
    for i in idx_order:
        vec = vecs[:, i]
        zb = vec[3 * (N + 1):4 * (N + 1)]
        sym = diag.calculate_symmetry_index(zb, N)
        if symmetric and sym < 0.2:
            return vec, vals[i]
        if (not symmetric) and sym > sym_threshold:
            return vec, vals[i]
    return None, None


def main():
    print("=== Landau coefficient sweep across beta and sigma ===")

    betas = [15, 20, 25, 30]
    sigmas = [0.0, 0.05, 0.10, 0.15, 0.20]

    Cf = 0.01
    Fr = 0.5
    N = 16

    # Natural subharmonic pairing: k_A = 2 k_B.
    k_A_val = 2.0
    k_B_val = 1.0

    D, y = diag.cheb_collocation(N)

    results = []

    for beta_val in betas:
        for sigma_val in sigmas:
            print(f"\nProcessing beta = {beta_val}, sigma = {sigma_val} ...")
            params = {
                "beta": beta_val,
                "Cf": Cf,
                "Fr": Fr,
                "transport_exponent_u": 3.0,
                "slope_coeff": 0.5,
                "Exner_Coeff": 0.005,
            }

            # 1. Central bar at k = 2.0.
            A_cen, B_cen = swe.assemble_swe_matrices(
                D, y, k_A_val,
                params["beta"], params["Cf"], params["Fr"],
                N_curv=0.0, sigma_width=sigma_val, params=params,
            )
            vals_cen, vecs_cen = la.eig(A_cen, B_cen)
            valid = np.isfinite(vals_cen) & (np.real(vals_cen) < 2.0)
            vals_cen = vals_cen[valid]
            vecs_cen = vecs_cen[:, valid]
            mode_A, omega_A = identify_mode(
                vecs_cen, vals_cen,
                np.argsort(np.real(vals_cen))[::-1], N, symmetric=True,
            )

            # 2. Alternate bar at k = 1.0.
            A_alt, B_alt = swe.assemble_swe_matrices(
                D, y, k_B_val,
                params["beta"], params["Cf"], params["Fr"],
                N_curv=0.0, sigma_width=sigma_val, params=params,
            )
            vals_alt, vecs_alt = la.eig(A_alt, B_alt)
            valid = np.isfinite(vals_alt) & (np.real(vals_alt) < 2.0)
            vals_alt = vals_alt[valid]
            vecs_alt = vecs_alt[:, valid]
            mode_B, omega_B = identify_mode(
                vecs_alt, vals_alt,
                np.argsort(np.real(vals_alt))[::-1], N, symmetric=False,
                sym_threshold=0.5,
            )

            if mode_A is None or mode_B is None:
                print(
                    f"Skipping: both modes not found for "
                    f"beta={beta_val}, sigma={sigma_val}"
                )
                continue

            # 3. Coefficients: intentional Table 2 single-carrier baseline (k_b=None -> k=1.0).
            calc = LandauCoupledCalculator(params, N, sigma_width=sigma_val)
            try:
                g_aa, g_ab, g_ba, g_bb = calc.compute_full_coefficients(
                    mode_A, omega_A, mode_B, omega_B, k_a=k_B_val, k_b=None
                )
                print(
                    f"gamma_AA={g_aa:.4f}, gamma_BB={g_bb:.4f}, "
                    f"gamma_AB={g_ab:.4f}, gamma_BA={g_ba:.4f}"
                )
                results.append({
                    "beta": beta_val,
                    "sigma": sigma_val,
                    "omega_A_real": np.real(omega_A),
                    "omega_A_imag": np.imag(omega_A),
                    "omega_B_real": np.real(omega_B),
                    "omega_B_imag": np.imag(omega_B),
                    "g_aa_real": np.real(g_aa),
                    "g_aa_imag": np.imag(g_aa),
                    "g_bb_real": np.real(g_bb),
                    "g_bb_imag": np.imag(g_bb),
                    "g_ab_real": np.real(g_ab),
                    "g_ab_imag": np.imag(g_ab),
                    "g_ba_real": np.real(g_ba),
                    "g_ba_imag": np.imag(g_ba),
                })
            except (la.LinAlgError, np.linalg.LinAlgError, ValueError) as exc:
                print(f"Coefficient evaluation failed: {exc}")

    if not results:
        raise RuntimeError("No parameter pairs produced coefficients.")

    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "landau_sweep_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved {len(results)} rows to {csv_path}")


if __name__ == "__main__":
    main()
