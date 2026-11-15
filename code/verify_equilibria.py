"""Independent verification of the regularized Landau equilibria.

The mixed state of the coupled amplitude system is checked here without reference
to any figure script.  Three things are verified:

1. **Exact equilibria.**  Setting the time derivatives to zero gives the coupled
   algebraic system for ``x = |A|^2`` and ``y = |B|^2``.  Eliminating ``y`` leaves
   a single quadratic in ``x``, whose positive root is evaluated in closed form
   rather than by iteration.
2. **Radial stability.**  The Jacobian of the modulus system is assembled at the
   equilibrium, and its trace, determinant, and eigenvalues are reported.  A
   stable node requires negative trace, positive determinant, and real negative
   eigenvalues.
3. **Regularization scaling.**  The strong-regularization fourth-root limit
   ``(c0 / g5)^(1/4)`` and the vanishing-regularization saturation
   ``sqrt(-c0 / c1)`` are compared against the swept values.

The rounded four-decimal coefficients of main-text Table 2 give
``|A_eq| = 20.42``, ``|B_eq| = 10.06``, and an amplification of ``5.83x``.  The
full-precision solver output stored in ``data/landau_sweep_results.csv`` gives
slightly larger values; both are reported.

Run from the repository root::

    python code/verify_equilibria.py
"""

import csv
import math
import os

import numpy as np

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_CODE_DIR)
DATA_DIR = os.path.join(REPO_ROOT, "data")


def solve_mixed_equilibrium(sigma_a, sigma_b, g_aa, g_bb, g_ab, g_ba, g5):
    """Solve the regularized amplitude system in closed form.

    The system is

    ``sigma_A - g_aa x - g_ab y - g5 x^2 = 0`` and
    ``sigma_B - g_bb y - g_ba x = 0``

    with ``x = |A|^2``, ``y = |B|^2``, and ``g_ij = Re(gamma_ij)``.  Substituting
    the second equation into the first yields
    ``g5 x^2 - d x - c = 0`` with
    ``c = sigma_A - g_ab sigma_B / g_bb`` and
    ``d = -(g_aa - g_ab g_ba / g_bb)``.

    Args:
        sigma_a, sigma_b: Real linear growth rates.
        g_aa, g_bb, g_ab, g_ba: Real parts of the Landau coefficients.
        g5: Fifth-order regularization coefficient.

    Returns:
        Dictionary with the discriminant, the squared and unsquared moduli, the
        isolated alternate-bar modulus, and the amplification factor.
    """
    c = sigma_a - (g_ab * sigma_b) / g_bb
    d = -(g_aa - (g_ab * g_ba) / g_bb)

    disc = d**2 + 4.0 * g5 * c
    if disc < 0:
        raise ValueError(f"Negative discriminant: {disc}")

    x = (d + np.sqrt(disc)) / (2.0 * g5)
    y = (sigma_b - g_ba * x) / g_bb

    r_a = np.sqrt(x) if x >= 0 else np.nan
    r_b = np.sqrt(y) if y >= 0 else np.nan

    y_iso = sigma_b / g_bb
    r_b_iso = np.sqrt(y_iso) if y_iso >= 0 else np.nan

    amp_ratio = r_b / r_b_iso if (r_b_iso > 0 and not np.isnan(r_b)) else np.nan

    return {
        "c": c,
        "d": d,
        "disc": disc,
        "x": x,
        "y": y,
        "R_A": r_a,
        "R_B": r_b,
        "R_B_iso": r_b_iso,
        "amplification": amp_ratio,
    }


def compute_radial_jacobian(sigma_a, sigma_b, g_aa, g_bb, g_ab, g_ba, g5, r_a, r_b):
    """Jacobian of the modulus system linearized at the positive equilibrium.

    The modulus dynamics are

    ``dR_A/dt = sigma_A R_A - g_aa R_A^3 - g_ab R_A R_B^2 - g5 R_A^5`` and
    ``dR_B/dt = sigma_B R_B - g_bb R_B^3 - g_ba R_B R_A^2``.

    Args:
        sigma_a, sigma_b: Real growth rates (unused at equilibrium, kept for API
            symmetry with the amplitude system).
        g_aa, g_bb, g_ab, g_ba: Real Landau coefficients.
        g5: Fifth-order coefficient.
        r_a, r_b: Positive equilibrium moduli.

    Returns:
        Dictionary with the Jacobian, its trace, determinant, sorted
        eigenvalues, and a stable-node flag.
    """
    x = r_a**2
    y = r_b**2

    # At equilibrium the explicit sigma terms cancel against the cubic balance.
    j11 = -2.0 * g_aa * x - 4.0 * g5 * (x**2)
    j12 = -2.0 * g_ab * r_a * r_b
    j21 = -2.0 * g_ba * r_a * r_b
    j22 = -2.0 * g_bb * y

    jac = np.array([[j11, j12], [j21, j22]])

    tr = j11 + j22
    det = j11 * j22 - j12 * j21

    eigvals = np.sort(np.linalg.eigvals(jac))

    return {
        "J": jac,
        "trace": tr,
        "det": det,
        "eigvals": eigvals,
        "is_stable_node": (
            tr < 0
            and det > 0
            and np.all(np.isreal(eigvals))
            and np.all(eigvals < 0)
        ),
    }


def read_reference_row(csv_path):
    """Return the ``beta = 15, sigma = 0.15`` row of the coefficient table.

    Args:
        csv_path: Path to ``landau_sweep_results.csv``.

    Returns:
        The matching row as a dictionary of floats, or ``None`` when absent.
    """
    if not os.path.exists(csv_path):
        return None
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (math.isclose(float(row["beta"]), 15.0)
                    and math.isclose(float(row["sigma"]), 0.15)):
                return {k: float(v) for k, v in row.items()}
    return None


def main():
    print("=" * 70)
    print("VERIFICATION OF THE REGULARIZED LANDAU EQUILIBRIA AND STABILITY")
    print("=" * 70)

    # Rounded four-decimal coefficients of main-text Table 2.
    sigma_a = 0.3085
    sigma_b = 0.1629
    g_aa = -0.0030
    g_bb = 0.0548
    g_ab = 0.0137
    g_ba = -0.0129

    print("\n--- Reference case (rounded Table 2 coefficients, g5 = 1e-6) ---")
    res_ref = solve_mixed_equilibrium(sigma_a, sigma_b, g_aa, g_bb, g_ab, g_ba,
                                      g5=1e-6)
    jac_ref = compute_radial_jacobian(
        sigma_a, sigma_b, g_aa, g_bb, g_ab, g_ba, g5=1e-6,
        r_a=res_ref["R_A"], r_b=res_ref["R_B"],
    )

    print(f"c = sigma_A - (g_AB sigma_B)/g_BB = {res_ref['c']:.6f}")
    print(f"d = -(g_AA - (g_AB g_BA)/g_BB)   = {res_ref['d']:.6f}")
    print(f"x = |A_eq|^2 = {res_ref['x']:.4f}  =>  |A_eq| = {res_ref['R_A']:.4f}")
    print(f"y = |B_eq|^2 = {res_ref['y']:.4f}  =>  |B_eq| = {res_ref['R_B']:.4f}")
    print(f"|B_iso|      = {res_ref['R_B_iso']:.4f}")
    print(f"Amplification = |B_eq|/|B_iso| = {res_ref['amplification']:.4f}x")

    print("\nJacobian at the reference equilibrium:")
    print(f"  J_11 = {jac_ref['J'][0, 0]:.4f}")
    print(f"  J_12 = {jac_ref['J'][0, 1]:.4f}")
    print(f"  J_21 = {jac_ref['J'][1, 0]:.4f}")
    print(f"  J_22 = {jac_ref['J'][1, 1]:.4f}")
    print(f"  Trace = {jac_ref['trace']:.4f}")
    print(f"  Det   = {jac_ref['det']:.4f}")
    print(f"  Eigenvalues: lambda_1 = {jac_ref['eigvals'][0]:.4f}, "
          f"lambda_2 = {jac_ref['eigvals'][1]:.4f}")
    print(f"  Stable radial node: {jac_ref['is_stable_node']}")

    # Sweep over the regularization coefficient.
    print("\n--- Sensitivity to g5 on [1e-8, 1e-4] ---")
    g5_list = [1.0e-8, 1.0e-7, 5.0e-7, 1.0e-6, 5.0e-6, 1.0e-5, 5.0e-5, 1.0e-4]
    print(f"{'g5':<10} | {'|A_eq|':<8} | {'|B_eq|':<8} | {'|B_iso|':<8} | "
          f"{'Amplification':<14} | {'Eig_1':<8} | {'Eig_2':<8}")
    print("-" * 80)
    for g5_val in g5_list:
        res = solve_mixed_equilibrium(sigma_a, sigma_b, g_aa, g_bb, g_ab, g_ba,
                                      g5=g5_val)
        jac = compute_radial_jacobian(
            sigma_a, sigma_b, g_aa, g_bb, g_ab, g_ba, g5=g5_val,
            r_a=res["R_A"], r_b=res["R_B"],
        )
        eig1, eig2 = jac["eigvals"][0], jac["eigvals"][1]
        eig1_str = (
            f"{eig1.real:.2f}{eig1.imag:+.2f}j" if np.iscomplex(eig1)
            else f"{eig1.real:.2f}"
        )
        eig2_str = (
            f"{eig2.real:.2f}{eig2.imag:+.2f}j" if np.iscomplex(eig2)
            else f"{eig2.real:.2f}"
        )
        print(f"{g5_val:<10.1e} | {res['R_A']:<8.2f} | {res['R_B']:<8.2f} | "
              f"{res['R_B_iso']:<8.2f} | {res['amplification']:<14.2f}x | "
              f"{eig1_str:<8} | {eig2_str:<8}")

    # Asymptotic limits.
    print("\n--- Asymptotic scaling limits ---")
    print(f"Strong-regularization limit (c/g5)^(1/4) at g5 = 1e-4: "
          f"{(res_ref['c'] / 1e-4)**0.25:.4f}")
    res_1e4 = solve_mixed_equilibrium(sigma_a, sigma_b, g_aa, g_bb, g_ab, g_ba,
                                      g5=1.0e-4)
    print(f"Swept mixed-branch |A_eq| at g5 = 1e-4: {res_1e4['R_A']:.4f}")
    print(f"Vanishing-regularization limit sqrt(-c/d): "
          f"{np.sqrt(-res_ref['c'] / res_ref['d']):.4f}")

    # Full-precision coefficients from the stored solver output.
    print("\n--- Full-precision coefficients from data/landau_sweep_results.csv ---")
    csv_path = os.path.join(DATA_DIR, "landau_sweep_results.csv")
    ref_row = read_reference_row(csv_path)

    if ref_row is None:
        print(f"Reference row not available in {csv_path}; skipping this check.")
        return

    res_raw = solve_mixed_equilibrium(
        ref_row["omega_A_real"], ref_row["omega_B_real"],
        ref_row["g_aa_real"], ref_row["g_bb_real"],
        ref_row["g_ab_real"], ref_row["g_ba_real"], g5=1e-6,
    )
    print(f"  sigma_A = {ref_row['omega_A_real']:.8f}, "
          f"sigma_B = {ref_row['omega_B_real']:.8f}")
    print(f"  g_aa = {ref_row['g_aa_real']:.8f}, g_bb = {ref_row['g_bb_real']:.8f}")
    print(f"  g_ab = {ref_row['g_ab_real']:.8f}, g_ba = {ref_row['g_ba_real']:.8f}")
    print("Full-precision equilibrium:")
    print(f"  |A_eq| = {res_raw['R_A']:.5f}")
    print(f"  |B_eq| = {res_raw['R_B']:.5f}")
    print(f"  |B_iso| = {res_raw['R_B_iso']:.5f}")
    print(f"  Amplification = {res_raw['amplification']:.5f} "
          f"=> rounds to {res_raw['amplification']:.2f}x")


if __name__ == "__main__":
    main()
