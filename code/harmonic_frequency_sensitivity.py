"""Sensitivity of the coupling coefficients to the harmonic operator choice.

The second-order problem admits two operator conventions:

* **Unshifted (quasi-steady).**  The sum harmonic solves
  ``A_2 X = F_2`` at the mixed wavenumber.  This is the standard
  morphodynamic approximation: the hydrodynamic adjustment time ``B_0/U_r`` is
  orders of magnitude shorter than the bed adjustment time, so the water waves
  pass through their steady state on the morphodynamic clock.
* **Frequency-shifted.**  The sum harmonic solves
  ``(A_2 - (lambda_A + lambda_B) B_2) X = F_2``, retaining the slow time
  dependence of the second-order envelope.

This script evaluates both conventions at the reference parameters
(``beta = 15.0``, ``sigma = 0.15``, ``k_A = 2.0``, ``k_B = 1.0``) and writes the
comparison to ``data/harmonic_frequency_sensitivity.csv``.  The result quantifies
how strongly the mixed sum-harmonic contribution depends on that operator
choice; the self-interaction coefficients are unaffected because the shift
vanishes for the ``omega_sum = None`` and self-harmonic solves at the same
wavenumber.

Run from the repository root::

    python code/harmonic_frequency_sensitivity.py
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

OMEGA_A_REF = 0.3085 - 5.9616j
OMEGA_B_REF = 0.1629 - 4.7062j


def compare_harmonic_solves(N=16, sigma_val=0.15):
    """Evaluate the four coefficients under both harmonic operator conventions.

    Args:
        N: Chebyshev polynomial degree.
        sigma_val: Width gradient.

    Returns:
        List of two row dictionaries, one per operator convention.
    """
    D, y = diag.cheb_collocation(N)
    params = {
        "beta": 15.0, "Cf": 0.01, "Fr": 0.5,
        "transport_exponent_u": 3.0, "slope_coeff": 0.5, "Exner_Coeff": 0.005,
    }

    A_a, B_a = swe.assemble_swe_matrices(
        D, y, 2.0, params["beta"], params["Cf"], params["Fr"],
        N_curv=0.0, sigma_width=sigma_val, params=params,
    )
    va, veca = la.eig(A_a, B_a)
    ma = np.isfinite(va) & (np.real(va) < 2.0)
    va, veca = va[ma], veca[:, ma]
    if not np.any(ma):
        raise ValueError("No valid central-bar eigenvalues found.")
    idx_a = np.argmin(np.abs(va - OMEGA_A_REF))
    mode_A, om_A = veca[:, idx_a], va[idx_a]

    A_b, B_b = swe.assemble_swe_matrices(
        D, y, 1.0, params["beta"], params["Cf"], params["Fr"],
        N_curv=0.0, sigma_width=sigma_val, params=params,
    )
    vb, vecb = la.eig(A_b, B_b)
    mb = np.isfinite(vb) & (np.real(vb) < 2.0)
    vb, vecb = vb[mb], vecb[:, mb]
    if not np.any(mb):
        raise ValueError("No valid alternate-bar eigenvalues found.")
    idx_b = np.argmin(np.abs(vb - OMEGA_B_REF))
    mode_B, om_B = vecb[:, idx_b], vb[idx_b]

    calc = LandauCoupledCalculator(params, N, sigma_width=sigma_val)

    # 1. Unshifted quasi-steady operator.
    res_un = calc.compute_full_coefficients(
        mode_A, om_A, mode_B, om_B, 2.0, 1.0, omega_sum=None
    )

    # 2. Frequency-shifted mixed sum harmonic.
    res_sh = calc.compute_full_coefficients(
        mode_A, om_A, mode_B, om_B, 2.0, 1.0, omega_sum=(om_A + om_B)
    )

    return [
        {
            "solver_type": "unshifted_quasi_steady",
            "omega_sum_applied": "None",
            "Re_gamma_AA": np.real(res_un[0]),
            "Re_gamma_AB": np.real(res_un[1]),
            "Re_gamma_BA": np.real(res_un[2]),
            "Re_gamma_BB": np.real(res_un[3]),
        },
        {
            "solver_type": "frequency_shifted",
            "omega_sum_applied": f"{om_A + om_B:.4f}",
            "Re_gamma_AA": np.real(res_sh[0]),
            "Re_gamma_AB": np.real(res_sh[1]),
            "Re_gamma_BA": np.real(res_sh[2]),
            "Re_gamma_BB": np.real(res_sh[3]),
        },
    ]


def main():
    try:
        rows = compare_harmonic_solves()
    except (ValueError, np.linalg.LinAlgError) as exc:
        print(f"Harmonic comparison failed: {exc}")
        raise

    os.makedirs(DATA_DIR, exist_ok=True)
    out_csv = os.path.join(DATA_DIR, "harmonic_frequency_sensitivity.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved harmonic frequency sensitivity to {out_csv}")
    print(
        f"  Unshifted:         Re(gamma_BA) = {rows[0]['Re_gamma_BA']:.6f}, "
        f"Re(gamma_AB) = {rows[0]['Re_gamma_AB']:.6f}"
    )
    print(
        f"  Frequency-shifted: Re(gamma_BA) = {rows[1]['Re_gamma_BA']:.6f}, "
        f"Re(gamma_AB) = {rows[1]['Re_gamma_AB']:.6f}"
    )


if __name__ == "__main__":
    main()
