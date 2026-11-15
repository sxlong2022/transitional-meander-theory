"""Sensitivity sweeps for the weakly nonlinear morphodynamic model.

Three sweeps probe the robustness of the cross-enhancement result at the
reference parameters ``beta = 15.0``, ``sigma = 0.15``:

1. **Fifth-order regularization** ``g_5 in [1e-8, 1e-4]``: re-solves the
   algebraic equilibrium system for the modal amplitudes and the alternate-bar
   amplification factor.
2. **Transverse bed-slope deflection** ``r_s in [0.3, 0.7]``: re-evaluates the
   linear eigenvalues and all four Landau coefficients, tracking each eigenmode
   continuously from the ``r_s = 0.5`` reference state.
3. **Wavenumber pairing**: compares natural subharmonic pairs
   (``k_A = 2 k_B``) against co-evolving single-carrier sets
   (``k_A = k_B = k_c``).

Outputs are written to ``data/sensitivity_g5.csv``, ``data/sensitivity_rs.csv``,
and ``data/sensitivity_wavenumbers.csv``.

The three sweeps share the tracking and projection machinery of
``landau_calculator`` and are kept in one file so the table provenance stays
auditable in a single place.
allow: SIZE_OK

Run from the repository root::

    python code/run_sensitivity_sweeps.py
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

# Reference eigenvalues anchoring continuous branch tracking.
OMEGA_A_REF = 0.3085 - 5.9616j
OMEGA_B_REF = 0.1629 - 4.7062j
OMEGA_B2_REF = 0.2083 - 7.0364j


def run_g5_sweep():
    """Sweep the fifth-order regularization coefficient on ``[1e-8, 1e-4]``.

    The mixed-state equilibrium solves

    ``sigma_A - g_aa x - g_ab y - g5 x^2 = 0`` and
    ``sigma_B - g_bb y - g_ba x = 0``

    with ``x = |A|^2`` and ``y = |B|^2``.  Substituting the second equation into
    the first leaves a monotone scalar root for ``x``, found here by Newton
    iteration.

    Returns:
        List of row dictionaries written to ``data/sensitivity_g5.csv``.
    """
    print("\n--- Sweep 1: g_5 sensitivity on [1e-8, 1e-4] ---")
    sigma_a = 0.3085
    sigma_b = 0.1629
    g_aa = -0.0030
    g_bb = 0.0548
    g_ab = 0.0137
    g_ba = -0.0129

    b_iso = np.sqrt(sigma_b / g_bb)

    g5_vals = [1e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4]
    results = []

    for g5 in g5_vals:
        r_a = 20.0
        converged = False
        for _ in range(100):
            r_a2 = r_a * r_a
            r_b2 = (sigma_b - g_ba * r_a2) / g_bb
            f = sigma_a - g_aa * r_a2 - g_ab * r_b2 - g5 * (r_a2 * r_a2)
            df = (-2 * g_aa * r_a
                  - g_ab * (-2 * g_ba * r_a / g_bb)
                  - 4 * g5 * (r_a2 * r_a))
            if abs(df) <= 1e-14:
                break
            r_a_next = r_a - f / df
            if abs(r_a_next - r_a) < 1e-12 and abs(f) < 1e-10:
                r_a = r_a_next
                converged = True
                break
            r_a = r_a_next

        r_b = np.sqrt((sigma_b - g_ba * (r_a**2)) / g_bb)
        amp_factor = r_b / b_iso

        if converged and np.isfinite(r_a) and r_a > 0:
            results.append({
                "g5": g5,
                "A_eq": r_a,
                "B_eq": r_b,
                "B_iso": b_iso,
                "amplification_factor": amp_factor,
            })
            print(
                f"  g_5={g5:.1e}: |A_eq|={r_a:.2f}, |B_eq|={r_b:.2f}, "
                f"amplification={amp_factor:.2f}x"
            )
        else:
            print(f"  g_5={g5:.1e}: Newton iteration did not converge.")

    out_csv = os.path.join(DATA_DIR, "sensitivity_g5.csv")
    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved g_5 sensitivity to {out_csv}")
    else:
        print("Warning: no g_5 results generated.")
    return results


def run_rs_sweep():
    """Sweep the transverse bed-slope deflection coefficient on ``[0.3, 0.7]``.

    Each eigenvalue problem is re-assembled for the new ``r_s``.  Branches are
    tracked continuously outward from ``r_s = 0.5`` in both directions, so the
    reported coefficients belong to the same physical eigenmodes throughout.

    Returns:
        List of row dictionaries written to ``data/sensitivity_rs.csv``.
    """
    print("\n--- Sweep 2: r_s transverse bed-slope sensitivity on [0.3, 0.7] ---")
    N = 16
    D, y = diag.cheb_collocation(N)

    rs_vals = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []

    params_ref = {
        "beta": 15.0, "Cf": 0.01, "Fr": 0.5,
        "transport_exponent_u": 3.0, "slope_coeff": 0.5, "Exner_Coeff": 0.005,
    }

    def modes_at(rs):
        """Return the tracked (A, B) eigenpairs at the given ``r_s`` from the anchor."""
        p = dict(params_ref)
        p["slope_coeff"] = rs
        A_a, B_a = swe.assemble_swe_matrices(
            D, y, 2.0, p["beta"], p["Cf"], p["Fr"],
            N_curv=0.0, sigma_width=0.15, params=p,
        )
        A_b, B_b = swe.assemble_swe_matrices(
            D, y, 1.0, p["beta"], p["Cf"], p["Fr"],
            N_curv=0.0, sigma_width=0.15, params=p,
        )
        return p, (A_a, B_a), (A_b, B_b)

    # Anchor at r_s = 0.5.
    p_ref, (A_a_ref, B_a_ref), (A_b_ref, B_b_ref) = modes_at(0.5)
    va_ref, veca_ref = la.eig(A_a_ref, B_a_ref)
    ma = np.isfinite(va_ref) & (np.real(va_ref) < 2.0)
    va_ref, veca_ref = va_ref[ma], veca_ref[:, ma]
    mode_A_ref = veca_ref[:, np.argmin(np.abs(va_ref - OMEGA_A_REF))]

    vb_ref, vecb_ref = la.eig(A_b_ref, B_b_ref)
    mb = np.isfinite(vb_ref) & (np.real(vb_ref) < 2.0)
    vb_ref, vecb_ref = vb_ref[mb], vecb_ref[:, mb]
    mode_B_ref = vecb_ref[:, np.argmin(np.abs(vb_ref - OMEGA_B_REF))]

    rs_results = {}

    # Backward: 0.5 -> 0.4 -> 0.3.
    cur_A, cur_B = mode_A_ref, mode_B_ref
    for rs in [0.5, 0.4, 0.3]:
        p, (A_a, B_a), (A_b, B_b) = modes_at(rs)
        va, veca = la.eig(A_a, B_a)
        ma = np.isfinite(va) & (np.real(va) < 2.0)
        va, veca = va[ma], veca[:, ma]
        vb, vecb = la.eig(A_b, B_b)
        mb = np.isfinite(vb) & (np.real(vb) < 2.0)
        vb, vecb = vb[mb], vecb[:, mb]
        if not np.any(ma) or not np.any(mb):
            continue

        idx_a = np.argmax(np.abs(veca.conj().T @ B_a @ cur_A))
        cur_A, om_A = veca[:, idx_a], va[idx_a]
        idx_b = np.argmax(np.abs(vecb.conj().T @ B_b @ cur_B))
        cur_B, om_B = vecb[:, idx_b], vb[idx_b]

        calc = LandauCoupledCalculator(p, N, sigma_width=0.15)
        rs_results[rs] = (om_A, om_B, calc.compute_full_coefficients(
            cur_A, om_A, cur_B, om_B, 2.0, 1.0
        ))

    # Forward: 0.5 -> 0.6 -> 0.7.
    cur_A, cur_B = mode_A_ref, mode_B_ref
    for rs in [0.5, 0.6, 0.7]:
        p, (A_a, B_a), (A_b, B_b) = modes_at(rs)
        va, veca = la.eig(A_a, B_a)
        ma = np.isfinite(va) & (np.real(va) < 2.0)
        va, veca = va[ma], veca[:, ma]
        vb, vecb = la.eig(A_b, B_b)
        mb = np.isfinite(vb) & (np.real(vb) < 2.0)
        vb, vecb = vb[mb], vecb[:, mb]
        if not np.any(ma) or not np.any(mb):
            continue

        idx_a = np.argmax(np.abs(veca.conj().T @ B_a @ cur_A))
        cur_A, om_A = veca[:, idx_a], va[idx_a]
        idx_b = np.argmax(np.abs(vecb.conj().T @ B_b @ cur_B))
        cur_B, om_B = vecb[:, idx_b], vb[idx_b]

        calc = LandauCoupledCalculator(p, N, sigma_width=0.15)
        rs_results[rs] = (om_A, om_B, calc.compute_full_coefficients(
            cur_A, om_A, cur_B, om_B, 2.0, 1.0
        ))

    for rs in rs_vals:
        if rs not in rs_results:
            print(f"  r_s={rs:.1f}: skipped, modes not found.")
            continue
        omega_A, omega_B, (g_aa, g_ab, g_ba, g_bb) = rs_results[rs]
        results.append({
            "r_s": rs,
            "lambda_A_real": np.real(omega_A),
            "lambda_A_imag": np.imag(omega_A),
            "lambda_B_real": np.real(omega_B),
            "lambda_B_imag": np.imag(omega_B),
            "g_aa_real": np.real(g_aa),
            "g_aa_imag": np.imag(g_aa),
            "g_bb_real": np.real(g_bb),
            "g_bb_imag": np.imag(g_bb),
            "g_ab_real": np.real(g_ab),
            "g_ab_imag": np.imag(g_ab),
            "g_ba_real": np.real(g_ba),
            "g_ba_imag": np.imag(g_ba),
        })
        print(
            f"  r_s={rs:.1f}: Re(omega_A)={np.real(omega_A):.4f}, "
            f"Re(omega_B)={np.real(omega_B):.4f}, "
            f"Re(gamma_BA)={np.real(g_ba):.6f}"
        )

    out_csv = os.path.join(DATA_DIR, "sensitivity_rs.csv")
    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved r_s sensitivity to {out_csv}")
    else:
        print("Warning: no r_s results generated.")
    return results


def run_wavenumber_sweep():
    """Sweep wavenumber pairings for subharmonic and co-evolving configurations.

    Subharmonic branches vary ``k_B`` on ``[0.5, 1.25]`` with ``k_A = 2 k_B`` at
    ``sigma = 0.15``.  Co-evolving branches set ``k_A = k_B = k_c`` and vary
    ``k_c`` on ``[1.5, 2.5]`` at ``sigma = 0.05``.  All branches are reached by
    incremental eigenvector tracking from the reference state so that the sampled
    coefficients belong to one continuous physical branch.

    Returns:
        List of row dictionaries written to ``data/sensitivity_wavenumbers.csv``.
    """
    print("\n--- Sweep 3: wavenumber pairing sensitivity ---")
    N = 16
    D, y = diag.cheb_collocation(N)
    params = {
        "beta": 15.0, "Cf": 0.01, "Fr": 0.5,
        "transport_exponent_u": 3.0, "slope_coeff": 0.5, "Exner_Coeff": 0.005,
    }

    def eig(k, sigma):
        """Return filtered, descending-sorted eigenpairs at ``(k, sigma)``."""
        A, B = swe.assemble_swe_matrices(
            D, y, k, params["beta"], params["Cf"], params["Fr"],
            N_curv=0.0, sigma_width=sigma, params=params,
        )
        v, vec = la.eig(A, B)
        m = np.isfinite(v) & (np.real(v) < 2.0)
        return A, B, v[m], vec[:, m]

    # Reference anchors at sigma = 0.15.
    _, B_a_ref, va_ref, veca_ref = eig(2.0, 0.15)
    mode_A_ref = veca_ref[:, np.argmin(np.abs(va_ref - OMEGA_A_REF))]
    om_A_ref = va_ref[np.argmin(np.abs(va_ref - OMEGA_A_REF))]

    _, B_b_ref, vb_ref, vecb_ref = eig(1.0, 0.15)
    mode_B_ref = vecb_ref[:, np.argmin(np.abs(vb_ref - OMEGA_B_REF))]
    om_B_ref = vb_ref[np.argmin(np.abs(vb_ref - OMEGA_B_REF))]

    _, B_b2_ref, vb2_ref, vecb2_ref = eig(2.0, 0.15)
    mode_B2_ref = vecb2_ref[:, np.argmin(np.abs(vb2_ref - OMEGA_B2_REF))]
    om_B2_ref = vb2_ref[np.argmin(np.abs(vb2_ref - OMEGA_B2_REF))]

    # Subharmonic branch, backward from k_B = 1.0 to 0.5.
    print("  Tracking subharmonic branch across k_B on [0.5, 1.25] ...")
    sub_tracked = {}
    cur_A, cur_B = mode_A_ref, mode_B_ref
    for kB_step in np.arange(1.0, 0.49, -0.05):
        kB_val = round(float(kB_step), 3)
        kA_val = round(2.0 * kB_val, 3)
        _, B_a, va, veca = eig(kA_val, 0.15)
        _, B_b, vb, vecb = eig(kB_val, 0.15)
        if va.size == 0 or vb.size == 0:
            continue
        idx_a = np.argmax(np.abs(veca.conj().T @ B_a @ cur_A))
        cur_A, cur_om_A = veca[:, idx_a], va[idx_a]
        idx_b = np.argmax(np.abs(vecb.conj().T @ B_b @ cur_B))
        cur_B, cur_om_B = vecb[:, idx_b], vb[idx_b]

        if kB_val in [1.0, 0.75, 0.5]:
            sub_tracked[kB_val] = (kA_val, kB_val, cur_A, cur_om_A, cur_B, cur_om_B)

    # Subharmonic branch, forward from k_B = 1.05 to 1.25.
    cur_A, cur_B = mode_A_ref, mode_B_ref
    for kB_step in np.arange(1.05, 1.26, 0.05):
        kB_val = round(float(kB_step), 3)
        kA_val = round(2.0 * kB_val, 3)
        _, B_a, va, veca = eig(kA_val, 0.15)
        _, B_b, vb, vecb = eig(kB_val, 0.15)
        if va.size == 0 or vb.size == 0:
            continue
        idx_a = np.argmax(np.abs(veca.conj().T @ B_a @ cur_A))
        cur_A, cur_om_A = veca[:, idx_a], va[idx_a]
        idx_b = np.argmax(np.abs(vecb.conj().T @ B_b @ cur_B))
        cur_B, cur_om_B = vecb[:, idx_b], vb[idx_b]

        if kB_val == 1.25:
            sub_tracked[kB_val] = (kA_val, kB_val, cur_A, cur_om_A, cur_B, cur_om_B)

    # Co-evolving branch down to sigma = 0.05 at k_c = 2.0.
    print("  Tracking co-evolving branch across sigma from 0.15 to 0.05 ...")
    cur_A, cur_B = mode_A_ref, mode_B2_ref
    cur_om_A = None
    cur_om_B = None
    for sig_step in np.arange(0.15, 0.049, -0.01):
        sig_val = round(float(sig_step), 3)
        _, B_a, va, veca = eig(2.0, sig_val)
        _, B_b, vb, vecb = eig(2.0, sig_val)
        if va.size == 0 or vb.size == 0:
            continue
        idx_a = np.argmax(np.abs(veca.conj().T @ B_a @ cur_A))
        cur_A, cur_om_A = veca[:, idx_a], va[idx_a]
        idx_b = np.argmax(np.abs(vecb.conj().T @ B_b @ cur_B))
        cur_B, cur_om_B = vecb[:, idx_b], vb[idx_b]

    if cur_om_A is None or cur_om_B is None:
        raise RuntimeError("Co-evolving branch tracking produced no valid modes.")
    co_tracked_005 = (2.0, 2.0, cur_A, cur_om_A, cur_B, cur_om_B)

    # Co-evolving branch across k_c at sigma = 0.05, backward to 1.5.
    cur_A_kc, cur_B_kc = co_tracked_005[2], co_tracked_005[4]
    mode_A_15, om_A_15, mode_B_15, om_B_15 = None, None, None, None
    for kc_step in np.arange(2.0, 1.49, -0.05):
        kc_val = round(float(kc_step), 3)
        _, B_a, va, veca = eig(kc_val, 0.05)
        _, B_b, vb, vecb = eig(kc_val, 0.05)
        if va.size == 0 or vb.size == 0:
            continue
        idx_a = np.argmax(np.abs(veca.conj().T @ B_a @ cur_A_kc))
        cur_A_kc, cur_om_A_kc = veca[:, idx_a], va[idx_a]
        idx_b = np.argmax(np.abs(vecb.conj().T @ B_b @ cur_B_kc))
        cur_B_kc, cur_om_B_kc = vecb[:, idx_b], vb[idx_b]
        if kc_val == 1.5:
            mode_A_15, om_A_15 = cur_A_kc, cur_om_A_kc
            mode_B_15, om_B_15 = cur_B_kc, cur_om_B_kc

    # Co-evolving branch across k_c at sigma = 0.05, forward to 2.5.
    cur_A_kc, cur_B_kc = co_tracked_005[2], co_tracked_005[4]
    mode_A_25, om_A_25, mode_B_25, om_B_25 = None, None, None, None
    for kc_step in np.arange(2.0, 2.51, 0.05):
        kc_val = round(float(kc_step), 3)
        _, B_a, va, veca = eig(kc_val, 0.05)
        _, B_b, vb, vecb = eig(kc_val, 0.05)
        if va.size == 0 or vb.size == 0:
            continue
        idx_a = np.argmax(np.abs(veca.conj().T @ B_a @ cur_A_kc))
        cur_A_kc, cur_om_A_kc = veca[:, idx_a], va[idx_a]
        idx_b = np.argmax(np.abs(vecb.conj().T @ B_b @ cur_B_kc))
        cur_B_kc, cur_om_B_kc = vecb[:, idx_b], vb[idx_b]
        if kc_val == 2.5:
            mode_A_25, om_A_25 = cur_A_kc, cur_om_A_kc
            mode_B_25, om_B_25 = cur_B_kc, cur_om_B_kc

    cases = []
    for kB_check in [0.5, 0.75, 1.0, 1.25]:
        if kB_check in sub_tracked:
            cfg = "subharmonic_ref" if kB_check == 1.0 else "subharmonic"
            entry = sub_tracked[kB_check]
            cases.append((cfg, entry[0], kB_check, 0.15,
                          entry[2], entry[3], entry[4], entry[5]))
    if mode_A_15 is not None and mode_B_15 is not None:
        cases.append(("co_evolving_window", 1.5, 1.5, 0.05,
                      mode_A_15, om_A_15, mode_B_15, om_B_15))
    if co_tracked_005[2] is not None and co_tracked_005[4] is not None:
        cases.append(("co_evolving_window", 2.0, 2.0, 0.05,
                      co_tracked_005[2], co_tracked_005[3],
                      co_tracked_005[4], co_tracked_005[5]))
    if mode_A_25 is not None and mode_B_25 is not None:
        cases.append(("co_evolving_window", 2.5, 2.5, 0.05,
                      mode_A_25, om_A_25, mode_B_25, om_B_25))
    if mode_A_ref is not None and mode_B2_ref is not None:
        cases.append(("co_evolving_diverging", 2.0, 2.0, 0.15,
                      mode_A_ref, om_A_ref, mode_B2_ref, om_B2_ref))

    results = []
    for config_type, kA, kB, sigma_val, mode_A, omega_A, mode_B, omega_B in cases:
        if mode_A is None or mode_B is None:
            continue
        calc = LandauCoupledCalculator(params, N, sigma_width=sigma_val)
        g_aa, g_ab, g_ba, g_bb = calc.compute_full_coefficients(
            mode_A, omega_A, mode_B, omega_B, kA, kB
        )
        results.append({
            "config": config_type,
            "sigma": sigma_val,
            "k_A": kA,
            "k_B": kB,
            "L_B_normalized": 2 * np.pi / kB,
            "lambda_A_real": np.real(omega_A),
            "lambda_B_real": np.real(omega_B),
            "g_aa_real": np.real(g_aa),
            "g_bb_real": np.real(g_bb),
            "g_ab_real": np.real(g_ab),
            "g_ba_real": np.real(g_ba),
            "cross_enhancement": bool(np.real(g_ba) < 0),
        })
        print(
            f"  {config_type} (sigma={sigma_val}): k_A={kA:.2f}, k_B={kB:.2f} "
            f"(L_B={2*np.pi/kB:.1f}) -> Re(gamma_BA)={np.real(g_ba):.6f} "
            f"(enhancement={np.real(g_ba) < 0})"
        )

    out_csv = os.path.join(DATA_DIR, "sensitivity_wavenumbers.csv")
    if results:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved wavenumber sensitivity to {out_csv}")
    else:
        print("Warning: no wavenumber results generated.")
    return results


def main():
    print("=" * 52)
    print("Parameter sensitivity sweeps for the coupled model")
    print("=" * 52)
    os.makedirs(DATA_DIR, exist_ok=True)
    run_g5_sweep()
    run_rs_sweep()
    run_wavenumber_sweep()
    print("\nAll sensitivity sweeps completed.")


if __name__ == "__main__":
    main()
