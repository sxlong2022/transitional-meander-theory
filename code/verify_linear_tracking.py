"""Audit of the linear growth-rate ordering across the width gradient.

Two parameter sets are examined side by side:

* **Reference set** ``beta = 15.0`` with ``Exner_Coeff = 0.005``, the parameters
  behind Table 2 and Figures 3 to 6.
* **Diagram set** ``beta = 10.0`` with the module defaults
  (``Exner_Coeff = 0.01``, ``slope_coeff = 0.5``), the parameters behind
  Figure 2.

For each set and each ``sigma`` the central-bar and alternate-bar branches are
tracked continuously from a known anchor, so the reported growth rates always
belong to the same physical eigenmodes.  At every step the script reports three
quality measures:

* the normalized overlap between consecutive eigenvectors,
* the runner-up overlap gap, which flags branch ambiguity,
* the relative eigenpair residual ``||A v - lambda B v||``.

A bracketed root solve then locates the width gradient at which the two growth
rates become equal.  The script asserts that the ordering and the growth or
damping state of both modes match the computed records, so a silent change in
solver defaults cannot pass unnoticed.

Note on scope: Figure 2 plots the maximum over a discrete wavenumber grid, which
is an envelope.  This audit holds ``k_A = 2.0`` and ``k_B = 1.0`` fixed and
therefore does not reproduce that envelope.

Run from the repository root::

    python code/verify_linear_tracking.py
"""

import os
import sys

import numpy as np
import scipy.linalg as la
from scipy.optimize import brentq

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import linear_solver as swe
import diagnostics as diag


def track_branch_with_diagnostics(k_val, target_ref, sigmas, params, N=16):
    """Track one eigenmode branch and record its continuation diagnostics.

    Args:
        k_val: Fixed wavenumber.
        target_ref: Anchor eigenvalue used to identify the branch at the first
            ``sigma``.
        sigmas: Width gradients in scan order.
        params: Physical parameter dictionary.
        N: Chebyshev polynomial degree.

    Returns:
        List of record dictionaries with ``sigma``, eigenvalue, overlap, gap, and
        relative residual.
    """
    D, y = diag.cheb_collocation(N)

    A0, B0 = swe.assemble_swe_matrices(
        D, y, k_val, params["beta"], params["Cf"], params["Fr"],
        0.0, sigmas[0], params=params,
    )
    v0, vec0 = la.eig(A0, B0)
    valid0 = np.isfinite(v0)
    v0 = v0[valid0]
    vec0 = vec0[:, valid0]

    idx0 = np.argmin(np.abs(v0 - target_ref))
    prev_vec = vec0[:, idx0] / np.linalg.norm(vec0[:, idx0])

    records = []

    for s_val in sigmas:
        A, B = swe.assemble_swe_matrices(
            D, y, k_val, params["beta"], params["Cf"], params["Fr"],
            0.0, s_val, params=params,
        )
        vals, vecs = la.eig(A, B)
        valid = np.isfinite(vals)
        vals = vals[valid]
        vecs = vecs[:, valid]

        norm_vecs = vecs / np.linalg.norm(vecs, axis=0, keepdims=True)
        overlaps = np.abs(np.conj(norm_vecs.T) @ prev_vec)

        sorted_indices = np.argsort(overlaps)[::-1]
        best_idx = sorted_indices[0]
        runner_up_idx = (
            sorted_indices[1] if len(sorted_indices) > 1 else best_idx
        )

        best_overlap = overlaps[best_idx]
        overlap_gap = best_overlap - overlaps[runner_up_idx]

        selected_val = vals[best_idx]
        selected_vec = vecs[:, best_idx]

        # Relative residual of the generalized eigenpair.
        res_vec = A @ selected_vec - selected_val * (B @ selected_vec)
        denom = (
            np.linalg.norm(A, 2) * np.linalg.norm(selected_vec)
            + np.abs(selected_val) * np.linalg.norm(B, 2)
            * np.linalg.norm(selected_vec)
        )
        rel_residual = np.linalg.norm(res_vec) / denom if denom > 0 else 0.0

        prev_vec = norm_vecs[:, best_idx]

        records.append({
            "sigma": s_val,
            "val": selected_val,
            "real": np.real(selected_val),
            "imag": np.imag(selected_val),
            "overlap": best_overlap,
            "gap": overlap_gap,
            "residual": rel_residual,
        })

    return records


def main():
    print("=" * 95)
    print("LINEAR STABILITY AUDIT: BRANCH TRACKING AND GROWTH-RATE ORDERING")
    print("=" * 95)

    params_b15 = {
        "beta": 15.0,
        "Cf": 0.01,
        "Fr": 0.5,
        "transport_exponent_u": 3.0,
        "slope_coeff": 0.5,
        "Exner_Coeff": 0.005,
    }

    sigmas = [0.15, 0.10, 0.05, 0.02, 0.00, -0.02, -0.05, -0.10, -0.15]

    # Unified tolerance for the regime classification of both parameter sets.
    DELTA_REGIME_TOL = 0.005

    target_A = 0.308477 - 5.961631j
    target_B = 0.162890 - 4.706235j

    rec_A_b15 = track_branch_with_diagnostics(2.0, target_A, sigmas, params_b15, N=16)
    rec_B_b15 = track_branch_with_diagnostics(1.0, target_B, sigmas, params_b15, N=16)

    print("\n--- 1. Reference set (beta = 15.0, k_A = 2.0, k_B = 1.0, "
          "Exner = 0.005) ---")
    print(f"{'sigma':<7} | {'Re(lambda_A)':<13} | {'Re(lambda_B)':<13} | "
          f"{'Delta':<11} | {'A: overlap (gap)':<18} | {'B: overlap (gap)':<18} "
          f"| Regime")
    print("-" * 115)

    for i, s in enumerate(sigmas):
        rA, rB = rec_A_b15[i], rec_B_b15[i]
        delta = rA["real"] - rB["real"]

        if delta > DELTA_REGIME_TOL:
            regime = "central (m=2) grows faster"
        elif delta < -DELTA_REGIME_TOL:
            regime = "alternate (m=1) decays slower"
        else:
            regime = "near-neutral transition"

        ov_A_str = f"{rA['overlap']:.3f} ({rA['gap']:.3f})"
        ov_B_str = f"{rB['overlap']:.3f} ({rB['gap']:.3f})"
        print(f"{s:<7.2f} | {rA['real']:<13.4f} | {rB['real']:<13.4f} | "
              f"{delta:<11.4f} | {ov_A_str:<18} | {ov_B_str:<18} | {regime}")

    # Diagram set: beta = 10.0 with explicit module defaults.
    params_b10 = {
        "beta": 10.0,
        "Cf": 0.01,
        "Fr": 0.5,
        "transport_exponent_u": 3.0,
        "slope_coeff": 0.5,
        "Exner_Coeff": 0.01,
    }

    target_A_b10 = 0.302417 - 5.961773j
    target_B_b10 = 0.157277 - 4.706507j
    rec_A_b10 = track_branch_with_diagnostics(2.0, target_A_b10, sigmas,
                                              params_b10, N=16)
    rec_B_b10 = track_branch_with_diagnostics(1.0, target_B_b10, sigmas,
                                              params_b10, N=16)

    print("\n--- 2. Diagram set (beta = 10.0, explicit defaults "
          "Exner = 0.01, r_s = 0.5) ---")
    print(f"{'sigma':<7} | {'Re(lambda_A)':<13} | {'Re(lambda_B)':<13} | "
          f"{'Delta':<11} | {'A: residual':<14} | {'B: residual':<14} | Regime")
    print("-" * 115)

    for i, s in enumerate(sigmas):
        rA, rB = rec_A_b10[i], rec_B_b10[i]
        delta = rA["real"] - rB["real"]

        if delta > DELTA_REGIME_TOL:
            regime = "central (m=2) grows faster"
        elif delta < -DELTA_REGIME_TOL:
            regime = "alternate (m=1) decays slower"
        else:
            regime = "near-neutral transition"

        print(f"{s:<7.2f} | {rA['real']:<13.4f} | {rB['real']:<13.4f} | "
              f"{delta:<11.4f} | {rA['residual']:<14.2e} | "
              f"{rB['residual']:<14.2e} | {regime}")

    # 3. Equal-damping crossing located on continuously tracked branches.
    # Selecting the nearest eigenvalue independently at each sigma is
    # discontinuous and makes brentq converge on a jump rather than a root.
    # Both branches are therefore tracked by eigenvector overlap across a fine
    # sigma grid, and the crossing is bracketed on the resulting curves.
    D16, y16 = diag.cheb_collocation(16)

    sig_fine = np.round(np.arange(0.15, -0.0501, -0.001), 4)
    rec_A_fine = track_branch_with_diagnostics(2.0, target_A, sig_fine,
                                               params_b15, N=16)
    rec_B_fine = track_branch_with_diagnostics(1.0, target_B, sig_fine,
                                               params_b15, N=16)

    delta_fine = np.array(
        [rA["real"] - rB["real"] for rA, rB in zip(rec_A_fine, rec_B_fine)]
    )
    sig_arr = np.array(sig_fine)

    sign_change = np.where(np.diff(np.sign(delta_fine)) != 0)[0]
    if sign_change.size == 0:
        raise ValueError("Tracked branches do not cross on the sampled range.")
    k = int(sign_change[0])

    def delta_interp(s):
        """Interpolate the tracked growth-rate difference at ``s``."""
        return float(np.interp(s, sig_arr[::-1], delta_fine[::-1]))

    s_root = brentq(delta_interp, sig_arr[k + 1], sig_arr[k], xtol=1e-10)
    res_root = delta_interp(s_root)

    gr_root_A = float(np.interp(s_root, sig_arr[::-1],
                                [r["real"] for r in rec_A_fine][::-1]))
    gr_root_B = float(np.interp(s_root, sig_arr[::-1],
                                [r["real"] for r in rec_B_fine][::-1]))

    print("\n--- 3. Equal-damping crossing (tracked-branch bracketing) ---")
    print(f"Root: sigma = {s_root:.6f} with Delta(sigma) = {res_root:.2e}")
    print(f"Growth rates at the root: Re(lambda_A) = {gr_root_A:.6f}, "
          f"Re(lambda_B) = {gr_root_B:.6f}")
    if abs(gr_root_A - gr_root_B) > 1e-4:
        raise AssertionError(
            f"Equal-damping claim violated: delta = {gr_root_A - gr_root_B:.2e}"
        )
    print(f"Both modes damped at the crossing: "
          f"{(gr_root_A < 0) and (gr_root_B < 0)}")

    # 4. Assert the ordering and growth/damping state against the records.
    def classify_checks(recs, label):
        """Verify ordering and growth state against the computed records."""
        checks = []
        for i, s in enumerate(sigmas):
            rA, rB = recs[0][i], recs[1][i]
            d = rA["real"] - rB["real"]
            if s >= 0.02:
                ok = (d > 0) and (rA["real"] > 0) and (rB["real"] > 0)
                checks.append((f"{label} sigma={s:+.2f}: Delta > 0, both growing", ok))
            elif s <= -0.02:
                ok = (d < 0) and (rA["real"] < 0) and (rB["real"] < 0)
                checks.append((f"{label} sigma={s:+.2f}: Delta < 0, both damped", ok))
        return checks

    all_checks = classify_checks((rec_A_b15, rec_B_b15), "beta=15")
    all_checks += classify_checks((rec_A_b10, rec_B_b10), "beta=10")

    print("\n" + "=" * 95)
    print("SYNTHESIS (verified against the computed records)")
    print("=" * 95)

    if not all_checks:
        raise RuntimeError("No ordering checks were generated; audit is vacuous.")

    failures = [c for c in all_checks if not c[1]]
    if failures:
        print("ASSERTION FAILURE: records contradict the stated ordering:")
        for name, _ in failures:
            print(f"  FAIL: {name}")
        raise AssertionError(f"{len(failures)} assertion(s) failed")

    print(f"All {len(all_checks)} assertions verified: ordering and growth state "
          f"match the computed records.")
    print()
    pos = [i for i, s in enumerate(sigmas) if s >= 0.02]
    neg = [i for i, s in enumerate(sigmas) if s <= -0.02]

    print("Reference set (beta = 15.0, Exner = 0.005):")
    print(f"  sigma in {[sigmas[i] for i in pos]}: both modes grow; Delta ranges "
          f"{min(rec_A_b15[i]['real'] - rec_B_b15[i]['real'] for i in pos):+.4f} "
          f"to {max(rec_A_b15[i]['real'] - rec_B_b15[i]['real'] for i in pos):+.4f}")
    print(f"  sigma in {[sigmas[i] for i in neg]}: both modes damped; Delta ranges "
          f"{min(rec_A_b15[i]['real'] - rec_B_b15[i]['real'] for i in neg):+.4f} "
          f"to {max(rec_A_b15[i]['real'] - rec_B_b15[i]['real'] for i in neg):+.4f}")
    print(f"  equal-damping root sigma = {s_root:.6f}, "
          f"Re(lambda_A) = {gr_root_A:.6f}, Re(lambda_B) = {gr_root_B:.6f}")
    print()
    print("Diagram set (beta = 10.0, Exner = 0.01):")
    print(f"  sigma in {[sigmas[i] for i in pos]}: central growth rate larger, "
          f"Delta {min(rec_A_b10[i]['real'] - rec_B_b10[i]['real'] for i in pos):+.4f} "
          f"to {max(rec_A_b10[i]['real'] - rec_B_b10[i]['real'] for i in pos):+.4f}")
    print(f"  sigma in {[sigmas[i] for i in neg]}: alternate decays more slowly, "
          f"Delta {min(rec_A_b10[i]['real'] - rec_B_b10[i]['real'] for i in neg):+.4f} "
          f"to {max(rec_A_b10[i]['real'] - rec_B_b10[i]['real'] for i in neg):+.4f}")


if __name__ == "__main__":
    main()
