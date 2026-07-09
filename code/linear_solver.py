"""Subproject-2: Linearized Shallow Water Equation (SWE) Stability Solver.

This module implements the Depth-Averaged (2D) stability analysis for
width-varying curved channels.

State Vector: X = [u, v, eta, zb]^T
Coordinates: (s, zeta), where zeta = n / b(s) in [-1, 1].
Resolution: Chebyshev spectral collocation in zeta.

Equations:
1. Streamwise Momentum (u)
2. Transverse Momentum (v)
3. Fluid Mass Conservation (eta)
4. Sediment Mass Conservation (zb, Exner)
"""

import numpy as np
import scipy.linalg as la
from typing import Dict, Tuple, Optional


def get_transport_coeffs(params: Dict) -> Dict[str, float]:
    """Get linearized sediment coefficients (Power Law / Parker)."""
    # Phi ~ u^M
    # Default M = 3 (for u^3 relation common in simpler models) or 2*1.5 (MPM)
    M = params.get("transport_exponent_u", 3.0)
    return {"Au": M, "Av": 0.0, "Ah": 0.0} # Av, Ah placeholder


def assemble_swe_matrices(
    D: np.ndarray,      # Chebyshev Diff Matrix (N+1, N+1)
    y: np.ndarray,      # Chebyshev Grid y in [-1, 1] (represents zeta)
    alpha: float,       # Wavenumber k (streamwise)
    beta: float,        # Aspect Ratio B/H
    Cf: float,          # Friction Coefficient
    Fr: float,          # Froude Number
    N_curv: float,      # Curvature parameter N = b * C
    sigma_width: float, # Width gradient parameter sigma = b'/b
    params: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble 4x4 Block Matrix for SWE + Exner.

    Dimensions:
    Each block is (M, M) where M = N_cheb + 1.
    Total Matrix Size: (4M, 4M).

    Row Ordering: u, v, eta, zb
    """
    M = D.shape[0]
    M4 = 4 * M

    A = np.zeros((M4, M4), dtype=complex)
    B = np.zeros((M4, M4), dtype=complex) # For eigenvalue: A X = omega B X

    # --- 1. Basic Operators ---
    I = np.eye(M)
    Zeta = np.diag(y) # Diagonal matrix of coordinate zeta

    # Transformed Longitudinal Derivative: d/ds = i*k - sigma * zeta * d/dzeta
    # Note: sigma_width = b'(s). transformation chain rule: d/ds|_n = d/ds|_z - (z*b'/b) d/dz
    # Let's assume input sigma_width is (b'/b).
    # Op_S = i * alpha * I - sigma_width * (Zeta @ D)
    Op_S = 1j * alpha * I - sigma_width * (Zeta @ D)

    # Transformed Transverse Derivative: d/dn = (1/b) * d/dzeta
    # We need b (half width).
    # If parameters are normalized such that b_0 = 1 (half width), then 1/b ~ 1.
    # We assume linearization around mean width b=1.
    Op_N = D # Assuming b=1 in non-dimensional units

    # --- 2. Geometry & Base Flow ---
    # Metric coefficient h_s = 1 + zeta * N_curv
    # If N_curv = 0 (straight), h_s = 1.
    hs_vec = 1.0 + y * N_curv
    Hs = np.diag(hs_vec)
    InvHs = np.diag(1.0 / hs_vec)

    # Base Flow U0 and H0 (D0)
    # For variable width, U0 * b * H0 = const.
    # We linearize around local section where b = 1, D0 = 1 (H0 = 1.0), and U0 = 1.0.
    U0_vec = 1.0 / hs_vec
    U0 = np.diag(U0_vec)
    H0 = 1.0  # Local depth normalized to 1.0

    # Non-uniform backwater flow depth gradient:
    # Ds' = (S0 - Cf/2) / (1 - Fr^2). We set bed slope S0 = Cf/2 as default (normal flow)
    # but allow user-specified bed_slope in params.
    S0 = params.get("bed_slope", 0.5 * Cf)
    Ds_prime = (S0 - 0.5 * Cf) / (1.0 - Fr**2)
    # Velocity gradient: Us' = - (sigma_width + Ds_prime)
    Us_prime = - (sigma_width + Ds_prime)

    # Correct Op_S with metric: d/ds_phys = (1/hs) * d/ds_xi
    Op_S_metric = InvHs @ Op_S

    # Friction Linearization: F_u = (Cf / H0) * U0
    F_u_vec = 2.0 * Cf * U0_vec
    F_u = np.diag(F_u_vec)

    # Transport Coefficients
    trans = get_transport_coeffs(params)
    Au = trans["Au"]

    # --- 3. Block Assembly Loop ---
    # Rows: 0:u, 1:v, 2:eta, 3:zb

    # Helper to slice blocks
    def blk(r, c):
        return (slice(r*M, (r+1)*M), slice(c*M, (c+1)*M))

    # -- Row 0: u-Momentum --
    # Term: - (U0/hs) * du/ds
    # Base Flow Gradient Correction:
    # Linearized Advection: u dU0/ds.
    # Continuity: d(U0 b)/ds = 0 -> dU0/ds = - U0 * b'/b = - U0 * sigma_width.
    # Moved to RHS (Matrix A): - u dU0/ds = + sigma U0 u.
    # Note: U0 is a matrix (diagonal), sigma is scalar.

    dU0_ds_term = (sigma_width + Ds_prime) * I

    A[blk(0,0)] = - U0 @ Op_S_metric - F_u + dU0_ds_term

    # Term: - g/hs * d(eta)/ds
    g_eff = 1.0 / (Fr**2)
    A[blk(0,2)] = - g_eff * InvHs @ Op_S + F_u

    # Term: - F * zb
    A[blk(0,3)] = - F_u

    # Term: Curvature/Coriolis-like + U0 * v / R_eff
    # Convention: - U0*C*v.
    C_val = N_curv # Assuming b=1
    A[blk(0,1)] = - U0 @ (C_val * InvHs) # - U0*C*v

    B[blk(0,0)] = I

    # -- Row 1: v-Momentum --
    # Centrifugal: + 2*U0*u / R  (Linearized u^2 -> 2 U0 u)

    # Advection v
    A[blk(1,1)] = - U0 @ Op_S_metric - F_u

    # Pressure
    A[blk(1,2)] = - g_eff * Op_N

    # Centrifugal Source: + 2 * U0 * C * u
    # Sign: Positive (Push to outer bank)
    A[blk(1,0)] = 2.0 * U0 @ (C_val * InvHs)

    B[blk(1,1)] = I

    # -- Row 2: Fluid Mass (eta) --
    # Div(u) term: - H0 * Div(u).
    # Div(u) in curved: 1/hs d(u)/ds + dv/dn + v/hs d(hs)/dn
    Div_s = InvHs @ Op_S
    Div_n = Op_N
    # Curvature divergence: v / hs * C
    Div_curv = InvHs * C_val

    A[blk(2,0)] = - H0 * Div_s - Ds_prime * I
    A[blk(2,1)] = - H0 * (Div_n + Div_curv)

    # Transport of surface: - U0/hs d(eta)/ds
    A[blk(2,2)] = - U0 @ Op_S_metric - Us_prime * I

    # Topographic steering: + U0/hs d(zb)/ds
    A[blk(2,3)] = U0 @ Op_S_metric + Us_prime * I

    B[blk(2,2)] = I

    # -- Row 3: Exner --
    # d(zb)/dt = - E_bed * [ d(Qs_s)/ds + d(Qs_n)/dy ]
    # Linearized Qs_s' = Au * u'
    # Linearized Qs_n' = v' - (r_s / beta) * d(zb')/dzeta
    E_bed = params.get("Exner_Coeff", 0.01)
    r_s = params.get("slope_coeff", 0.5)
    D2 = D @ D

    # streamwise sediment advection: - E_bed * Au * (d/ds - zeta * sigma * d/dzeta) u'
    A[blk(3,0)] = - E_bed * Au * Op_S_metric

    # transverse sediment advection: - E_bed * d(v')/dzeta
    A[blk(3,1)] = - E_bed * D

    # transverse bed slope effect: + E_bed * (r_s / beta) * d2(zb')/dzeta2
    A[blk(3,3)] = E_bed * (r_s / beta) * D2

    B[blk(3,3)] = I

    # --- 4. Boundary Conditions ---
    # Implement bank boundary conditions at zeta = +/- 1 (Indices 0 and M-1)

    # 4.1 Flow Boundary Condition: v' = +/- sigma * u'
    # Upper bank (zeta = 1, index 0)
    glob_r_up_v = 1 * M + 0
    A[glob_r_up_v, :] = 0.0
    B[glob_r_up_v, :] = 0.0
    A[glob_r_up_v, glob_r_up_v] = 1.0
    A[glob_r_up_v, 0 * M + 0] = -sigma_width

    # Lower bank (zeta = -1, index M-1)
    glob_r_down_v = 1 * M + M - 1
    A[glob_r_down_v, :] = 0.0
    B[glob_r_down_v, :] = 0.0
    A[glob_r_down_v, glob_r_down_v] = 1.0
    A[glob_r_down_v, 0 * M + M - 1] = sigma_width

    # 4.2 Sediment Boundary Condition: q_n' = +/- sigma * q_s'
    # q_n' = v' - (r_s / beta) * d(zb')/dzeta, q_s' = Au * u'
    # Upper bank (zeta = 1, index 0)
    glob_r_up_zb = 3 * M + 0
    A[glob_r_up_zb, :] = 0.0
    B[glob_r_up_zb, :] = 0.0
    A[glob_r_up_zb, 3 * M : 4 * M] = (r_s / beta) * D[0, :]
    A[glob_r_up_zb, 1 * M + 0] = -1.0
    A[glob_r_up_zb, 0 * M + 0] = Au * sigma_width

    # Lower bank (zeta = -1, index M-1)
    glob_r_down_zb = 3 * M + M - 1
    A[glob_r_down_zb, :] = 0.0
    B[glob_r_down_zb, :] = 0.0
    A[glob_r_down_zb, 3 * M : 4 * M] = (r_s / beta) * D[M-1, :]
    A[glob_r_down_zb, 1 * M + M - 1] = -1.0
    A[glob_r_down_zb, 0 * M + M - 1] = -Au * sigma_width
    return A, B
