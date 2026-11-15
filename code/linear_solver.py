"""Depth-averaged two-dimensional linear stability solver for variable-width channels.

The perturbation state vector is ``X = [u, v, eta, zb]^T``, where ``u`` and ``v``
are the streamwise and transverse velocity perturbations, ``eta`` is the free
surface elevation, and ``zb`` is the bed elevation.  The transverse coordinate is
stretched as ``zeta = n / b(s)`` on ``[-1, 1]``, so a fixed Chebyshev collocation
grid resolves a channel whose half-width ``b`` varies with the streamwise
coordinate ``s``.

The four governing equations are

1. streamwise momentum (``u``),
2. transverse momentum (``v``),
3. fluid mass conservation (``eta``),
4. sediment mass conservation (``zb``, Exner).

``assemble_swe_matrices`` returns the pencil ``(A, B)`` of the generalized
eigenvalue problem ``A X = omega B X``.  The real part of ``omega`` is the linear
growth rate and the imaginary part is ``-omega_0``, the bar migration frequency.
"""

import numpy as np
from typing import Dict, Tuple


def cheb_collocation(N: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return the Chebyshev differentiation matrix and its collocation nodes.

    Args:
        N: Polynomial degree; the grid has ``N + 1`` nodes.

    Returns:
        ``(D, x)`` where ``D`` is the ``(N + 1, N + 1)`` differentiation matrix
        and ``x`` are the nodes ordered from ``+1`` to ``-1``.
    """
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack(([2], np.ones(N - 1), [2])) * (-1) ** np.arange(N + 1)
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x


def get_transport_coeffs(params: Dict) -> Dict[str, float]:
    """Return the linearized bedload coefficients of the power-law closure."""
    M = params.get("transport_exponent_u", 3.0)
    return {"Au": M, "Av": 0.0, "Ah": 0.0}


def assemble_swe_matrices(
    D: np.ndarray,       # Chebyshev differentiation matrix, (N+1, N+1)
    y: np.ndarray,       # Chebyshev nodes in [-1, 1], representing zeta
    alpha: float,        # streamwise wavenumber k
    beta: float,         # aspect ratio B/H
    Cf: float,           # friction coefficient
    Fr: float,           # Froude number
    N_curv: float,       # curvature parameter N = b * C
    sigma_width: float,  # width gradient sigma = (1/b) db/ds
    params: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble the 4x4 block SWE plus Exner operator.

    Each block is ``(M, M)`` with ``M = N + 1``, so the assembled matrices are
    ``(4M, 4M)``.  Rows are ordered ``u, v, eta, zb``.

    Returns:
        ``(A, B)`` such that ``A X = omega B X``.
    """
    M = D.shape[0]
    M4 = 4 * M

    A = np.zeros((M4, M4), dtype=complex)
    B = np.zeros((M4, M4), dtype=complex)

    # --- 1. Basic operators ---
    I = np.eye(M)
    Zeta = np.diag(y)

    # Streamwise derivative in the stretched frame:
    #   d/ds = i*k - sigma * zeta * d/dzeta
    Op_S = 1j * alpha * I - sigma_width * (Zeta @ D)

    # Base-flow advection.  With V0 = zeta * b * sigma * U0 the metric advection
    # term cancels, so advection reduces to U0 * d/ds|_zeta when the flag is set.
    Op_S_advect = (
        1j * alpha * I if params.get("cancel_advect_metric", False) else Op_S.copy()
    )
    Op_N = D  # b = 1 in the non-dimensional reference section

    # --- 2. Geometry and base flow ---
    # Metric coefficient h_s = 1 + zeta * N_curv; straight channel gives h_s = 1.
    hs_vec = 1.0 + y * N_curv
    InvHs = np.diag(1.0 / hs_vec)

    U0_vec = 1.0 / hs_vec
    U0 = np.diag(U0_vec)
    H0 = 1.0  # local depth normalized to unity

    # Backwater depth gradient: (1 - Fr^2) dD0/ds = S0 - Cf/2 + Fr^2 sigma
    S0 = params.get("bed_slope", 0.5 * Cf)
    include_width_backwater = params.get("include_width_backwater", False)
    if np.isclose(Fr, 1.0):
        raise ValueError(
            "Transcritical flow (Fr = 1) is singular in the 1D backwater balance."
        )
    denom = 1.0 - Fr**2
    if include_width_backwater:
        Ds_prime = ((S0 - 0.5 * Cf) + (Fr**2) * sigma_width) / denom
    else:
        Ds_prime = (S0 - 0.5 * Cf) / denom

    # Velocity gradient from discharge conservation: Us' = -(sigma + Ds')
    Us_prime = -(sigma_width + Ds_prime)

    Op_S_metric = InvHs @ Op_S
    Op_S_advect_metric = InvHs @ Op_S_advect

    # Friction linearization: F_u = 2 Cf U0
    F_u = np.diag(2.0 * Cf * U0_vec)

    Au = get_transport_coeffs(params)["Au"]

    def blk(r, c):
        return (slice(r * M, (r + 1) * M), slice(c * M, (c + 1) * M))

    # -- Row 0: u-momentum --
    # Continuity of U0 * b gives dU0/ds = -U0 * sigma; moved to the RHS the
    # base-flow gradient contributes +(sigma + Ds') u.
    dU0_ds_term = (sigma_width + Ds_prime) * I

    A[blk(0, 0)] = -U0 @ Op_S_advect_metric - F_u + dU0_ds_term

    g_eff = 1.0 / (Fr**2)
    A[blk(0, 2)] = -g_eff * InvHs @ Op_S + F_u

    A[blk(0, 3)] = -F_u

    C_val = N_curv  # b = 1
    A[blk(0, 1)] = -U0 @ (C_val * InvHs)

    B[blk(0, 0)] = I

    # -- Row 1: v-momentum --
    A[blk(1, 1)] = -U0 @ Op_S_advect_metric - F_u

    A[blk(1, 2)] = -g_eff * Op_N

    # Centrifugal source: linearized u^2 -> 2 U0 u, pushing toward the outer bank.
    A[blk(1, 0)] = 2.0 * U0 @ (C_val * InvHs)

    B[blk(1, 1)] = I

    # -- Row 2: fluid mass (eta) --
    Div_s = InvHs @ Op_S
    Div_n = Op_N
    Div_curv = InvHs * C_val

    A[blk(2, 0)] = -H0 * Div_s - Ds_prime * I
    A[blk(2, 1)] = -H0 * (Div_n + Div_curv)

    A[blk(2, 2)] = -U0 @ Op_S_metric - Us_prime * I
    A[blk(2, 3)] = U0 @ Op_S_metric + Us_prime * I

    B[blk(2, 2)] = I

    # -- Row 3: Exner --
    # dzb/dt = -E_bed [d(Qs_s)/ds + d(Qs_n)/dy] with
    # Qs_s' = Au u' and Qs_n' = v' - (r_s/beta) dzb'/dzeta.
    E_bed = params.get("Exner_Coeff", 0.01)
    r_s = params.get("slope_coeff", 0.5)
    D2 = D @ D

    A[blk(3, 0)] = -E_bed * Au * Op_S_metric
    A[blk(3, 1)] = -E_bed * D
    A[blk(3, 3)] = E_bed * (r_s / beta) * D2

    B[blk(3, 3)] = I

    # --- 4. Boundary conditions at zeta = +/- 1 (indices 0 and M-1) ---
    # 4.1 Flow: v' = +/- sigma * u' at the two banks.
    glob_r_up_v = 1 * M + 0
    A[glob_r_up_v, :] = 0.0
    B[glob_r_up_v, :] = 0.0
    A[glob_r_up_v, glob_r_up_v] = 1.0
    A[glob_r_up_v, 0 * M + 0] = -sigma_width

    glob_r_down_v = 1 * M + M - 1
    A[glob_r_down_v, :] = 0.0
    B[glob_r_down_v, :] = 0.0
    A[glob_r_down_v, glob_r_down_v] = 1.0
    A[glob_r_down_v, 0 * M + M - 1] = sigma_width

    # 4.2 Sediment: linearizing q_y/q_s = V/U - (r_s / beta*b) dzb/dzeta about
    # V0/U0 = +/- b_s and q_y0/q_s0 = +/- b_s with v' = +/- b_s u' gives
    # dzb'/dzeta = 0 identically at both banks.
    glob_r_up_zb = 3 * M + 0
    A[glob_r_up_zb, :] = 0.0
    B[glob_r_up_zb, :] = 0.0

    glob_r_down_zb = 3 * M + M - 1
    A[glob_r_down_zb, :] = 0.0
    B[glob_r_down_zb, :] = 0.0

    if params.get("legacy_sediment_bc", True):
        # Incomplete linearization retained for backward-compatible eigenvalues.
        A[glob_r_up_zb, 3 * M : 4 * M] = (r_s / beta) * D[0, :]
        A[glob_r_up_zb, 1 * M + 0] = -1.0
        A[glob_r_up_zb, 0 * M + 0] = Au * sigma_width

        A[glob_r_down_zb, 3 * M : 4 * M] = (r_s / beta) * D[M - 1, :]
        A[glob_r_down_zb, 1 * M + M - 1] = -1.0
        A[glob_r_down_zb, 0 * M + M - 1] = -Au * sigma_width
    else:
        # Exact sediment impermeability at both banks.
        A[glob_r_up_zb, 3 * M : 4 * M] = D[0, :]
        A[glob_r_down_zb, 3 * M : 4 * M] = D[M - 1, :]

    return A, B
