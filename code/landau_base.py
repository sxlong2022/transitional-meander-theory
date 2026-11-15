"""Weakly nonlinear amplitude-equation machinery for a single bar mode.

The linear stability operator in :mod:`linear_solver` is augmented here with the
second-order forced responses needed by the Stuart--Landau reduction.  For a
perturbation expansion in the small parameter ``epsilon``, the quadratic
nonlinearities of the shallow water plus Exner system produce

* a zero-wavenumber mean-flow distortion driven by the Reynolds stresses of the
  fundamental mode against its complex conjugate, and
* a second harmonic at twice the carrier wavenumber driven by self-interaction
  of the fundamental.

Both responses are non-secular and are obtained by solving the linear operator
at the appropriate output wavenumber.  The cubic Landau coefficient then follows
from projecting the third-order forcing onto the adjoint of the fundamental mode.

The class here supplies the grid, the direct and adjoint eigenproblems, and the
forcing assemblers.  :mod:`landau_calculator` adds the coupled two-mode
projection that produces the four coefficients of the amplitude system.
"""

import os
import sys

import numpy as np
import scipy.linalg as la

# Keep the module importable both as ``code.landau_base`` and as a direct script.
_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from linear_solver import assemble_swe_matrices


class LandauCalculator:
    """Grid, eigensolver, and second-order forcing assembler for one bar mode."""

    def __init__(self, params, N, L_domain=None):
        """Initialize the calculator.

        Args:
            params: Physical parameter dictionary (``beta``, ``Cf``, ``Fr``,
                ``transport_exponent_u``, ``slope_coeff``, ``Exner_Coeff``).
            N: Chebyshev polynomial degree; the grid holds ``N + 1`` nodes.
            L_domain: Optional physical domain length, retained for callers that
                normalize amplitudes by wavelength.
        """
        self.params = params
        self.N = N
        self.L_domain = L_domain
        self.y_cheb = np.cos(np.pi * np.arange(N + 1) / N)
        self.D_cheb = self.cheb_diff_matrix(N)

        self.eigenvalue_linear = None
        self.vector_linear = None
        self.vector_adjoint = None

    def cheb_diff_matrix(self, N):
        """Return the Chebyshev differentiation matrix on ``N + 1`` nodes."""
        x = np.cos(np.pi * np.arange(N + 1) / N)
        c = np.hstack(([2], np.ones(N - 1), [2])) * (-1) ** np.arange(N + 1)
        X = np.tile(x, (N + 1, 1))
        dX = X - X.T
        D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(N + 1))
        D = D - np.diag(np.sum(D, axis=1))
        return D

    def solve_linear_adjoint(self, sigma_width, k_wavenumber):
        """Solve the direct and adjoint eigenproblems for the most unstable mode.

        Args:
            sigma_width: Width gradient ``sigma = (1/b) db/ds``.
            k_wavenumber: Streamwise wavenumber ``k``.

        Returns:
            The complex eigenvalue of the selected direct mode.
        """
        A, B = assemble_swe_matrices(
            self.D_cheb,
            self.y_cheb,
            alpha=k_wavenumber,
            beta=self.params.get("beta", 10.0),
            Cf=self.params.get("Cf", 0.01),
            Fr=self.params.get("Fr", 0.5),
            N_curv=0.0,
            sigma_width=sigma_width,
            params=self.params,
        )

        vals, vecs = la.eig(A, B)

        # Algebraic bank constraints make part of the spectrum infinite.
        finite_mask = np.isfinite(vals)
        vals_finite = vals[finite_mask]
        vecs_finite = vecs[:, finite_mask]

        if len(vals_finite) == 0:
            raise ValueError("No finite eigenvalues found.")

        idx = np.argmax(np.real(vals_finite))
        self.eigenvalue_linear = vals_finite[idx]
        self.vector_linear = vecs_finite[:, idx]

        # Adjoint problem: A^H Y = conj(omega) B^H Y.
        vals_adj, vecs_adj = la.eig(A.conj().T, B.conj().T)

        target = np.conj(self.eigenvalue_linear)
        idx_adj = np.argmin(np.abs(vals_adj - target))
        self.vector_adjoint = vecs_adj[:, idx_adj]

        # Normalize so that <psi_adj, B psi_direct> = 1.
        norm_factor = np.vdot(self.vector_adjoint, B @ self.vector_linear)
        self.vector_adjoint = self.vector_adjoint / np.conj(norm_factor)

        return self.eigenvalue_linear

    def extract_components(self, vector):
        """Split a state vector into ``(u, v, eta, zb)`` blocks."""
        M = self.N + 1
        return vector[0:M], vector[M:2 * M], vector[2 * M:3 * M], vector[3 * M:4 * M]

    def compute_nonlinear_forcing(self, k_wavenumber):
        """Assemble the second-order forcing vectors of the fundamental mode.

        Args:
            k_wavenumber: Carrier wavenumber ``k``.

        Returns:
            ``(forcing_20, forcing_22)``.  ``forcing_20`` is the zero-wavenumber
            mean-flow forcing driven by the Reynolds stresses; ``forcing_22`` is
            the second-harmonic forcing at ``2k`` driven by self-interaction.
        """
        if self.vector_linear is None:
            raise ValueError("Linear mode not solved yet.")

        u1, v1, eta1, zb1 = self.extract_components(self.vector_linear)
        u1_conj = np.conj(u1)
        v1_conj = np.conj(v1)

        D = self.D_cheb
        ik = 1j * k_wavenumber

        # --- Second harmonic (2k) ---
        u1_x = ik * u1
        u1_y = D @ u1
        v1_x = ik * v1
        v1_y = D @ v1

        N_u_22 = -(u1 * u1_x + v1 * u1_y)

        C_val = 0.0
        N_v_22 = -(u1 * v1_x + v1 * v1_y + C_val * u1**2)

        h1 = eta1 - zb1
        flux_u = h1 * u1
        flux_v = h1 * v1
        N_eta_22 = -(ik * flux_u + D @ flux_v)

        p_exp = self.params.get("transport_exponent_u", 3.0)
        N_zb_22 = -(p_exp * (p_exp - 1) * u1 * u1_x)
        forcing_22 = np.concatenate([N_u_22, N_v_22, N_eta_22, N_zb_22])

        # --- Mean flow (k = 0) ---
        uu_mean = 2 * np.real(u1 * u1_conj)
        vv_mean = 2 * np.real(v1 * v1_conj)
        uv_mean = 2 * np.real(u1 * v1_conj)

        N_u_20 = -(D @ uv_mean)
        N_v_20 = -(D @ vv_mean) - (C_val * uu_mean)

        hv_mean = 2 * np.real(h1 * v1_conj)
        N_eta_20 = -(D @ hv_mean)

        # Axisymmetric base state gives no streamwise divergence at k = 0.
        N_zb_20 = np.zeros_like(u1)

        forcing_20 = np.concatenate([N_u_20, N_v_20, N_eta_20, N_zb_20])

        return forcing_20, forcing_22

    def compute_nonlinear_forcing_mean_isolated(self, vec_a, vec_b, k):
        """Mean-flow forcing driven by the interaction of ``vec_a`` and ``vec_b*``.

        Args:
            vec_a: First state vector (typically the mode itself).
            vec_b: Second state vector, conjugated internally.
            k: Wavenumber of both vectors.

        Returns:
            Forcing vector of length ``4(N + 1)`` with zero bed block.
        """
        u_a, v_a, eta_a, _ = self.extract_components(vec_a)
        u_b, v_b, eta_b, _ = self.extract_components(vec_b)
        u_b = u_b.conj()
        v_b = v_b.conj()
        eta_b = eta_b.conj()

        adv_u_1 = u_a * (-1j * k * u_b) + v_a * (self.D_cheb @ u_b)
        adv_u_2 = u_b * (1j * k * u_a) + v_b * (self.D_cheb @ u_a)
        forcing_u = -(adv_u_1 + adv_u_2)

        adv_v_1 = u_a * (-1j * k * v_b) + v_a * (self.D_cheb @ v_b)
        adv_v_2 = u_b * (1j * k * v_a) + v_b * (self.D_cheb @ v_a)
        forcing_v = -(adv_v_1 + adv_v_2)

        flux_x = u_a * eta_b + u_b * eta_a
        flux_y = v_a * eta_b + v_b * eta_a
        forcing_h = -(0 + self.D_cheb @ flux_y)

        M = self.N + 1
        F = np.zeros(4 * M, dtype=complex)
        F[0:M] = forcing_u
        F[M:2 * M] = forcing_v
        F[2 * M:3 * M] = forcing_h
        return F

    def compute_nonlinear_forcing_harmonic_mixed(self, vec_a, vec_b, k):
        """Sum-harmonic forcing at ``2k`` for two vectors sharing wavenumber ``k``.

        Args:
            vec_a: First state vector.
            vec_b: Second state vector.
            k: Wavenumber of both vectors.

        Returns:
            Forcing vector of length ``4(N + 1)`` with zero bed block.
        """
        u_a, v_a, eta_a, _ = self.extract_components(vec_a)
        u_b, v_b, eta_b, _ = self.extract_components(vec_b)

        term1 = u_a * (1j * k * u_b) + v_a * (self.D_cheb @ u_b)
        term2 = u_b * (1j * k * u_a) + v_b * (self.D_cheb @ u_a)
        forcing_u = -(term1 + term2)

        term1_v = u_a * (1j * k * v_b) + v_a * (self.D_cheb @ v_b)
        term2_v = u_b * (1j * k * v_a) + v_b * (self.D_cheb @ v_a)
        forcing_v = -(term1_v + term2_v)

        flux_x = u_a * eta_b + u_b * eta_a
        flux_y = v_a * eta_b + v_b * eta_a
        div = 1j * (2 * k) * flux_x + self.D_cheb @ flux_y
        forcing_h = -div

        M = self.N + 1
        F = np.zeros(4 * M, dtype=complex)
        F[0:M] = forcing_u
        F[M:2 * M] = forcing_v
        F[2 * M:3 * M] = forcing_h
        return F

    def compute_cubic_interaction_mixed(self, vec_lin, vec_sec, k_lin, k_sec):
        """Cubic forcing from the interaction of a linear mode and a second-order mode.

        Args:
            vec_lin: Linear-mode state vector.
            vec_sec: Second-order state vector.
            k_lin: Wavenumber of ``vec_lin``.
            k_sec: Wavenumber of ``vec_sec``.

        Returns:
            Forcing vector at the summed wavenumber, zero in the bed block.
        """
        u1, v1, eta1, _ = self.extract_components(vec_lin)
        u2, v2, eta2, _ = self.extract_components(vec_sec)

        adv_u_12 = u1 * (1j * k_sec * u2) + v1 * (self.D_cheb @ u2)
        adv_u_21 = u2 * (1j * k_lin * u1) + v2 * (self.D_cheb @ u1)
        f_u = -(adv_u_12 + adv_u_21)

        adv_v_12 = u1 * (1j * k_sec * v2) + v1 * (self.D_cheb @ v2)
        adv_v_21 = u2 * (1j * k_lin * v1) + v2 * (self.D_cheb @ v1)
        f_v = -(adv_v_12 + adv_v_21)

        flux_x = u1 * eta2 + u2 * eta1
        flux_y = v1 * eta2 + v2 * eta1
        k_out = k_lin + k_sec
        f_h = -(1j * k_out * flux_x + self.D_cheb @ flux_y)

        M = self.N + 1
        F = np.zeros(4 * M, dtype=complex)
        F[0:M] = f_u
        F[M:2 * M] = f_v
        F[2 * M:3 * M] = f_h
        return F

    def compute_cubic_interaction(self, u1, v1, eta1, zb1, X20, X22, k_c):
        """Third-order forcing of the fundamental from mean and second-harmonic fields.

        Args:
            u1, v1, eta1, zb1: Fundamental-mode components.
            X20: Mean-flow response.
            X22: Second-harmonic response.
            k_c: Fundamental carrier wavenumber.

        Returns:
            Forcing vector at the carrier wavenumber, with only the streamwise
            momentum block populated.
        """
        u20, v20, _, _ = self.extract_components(X20)
        u22, v22, _, _ = self.extract_components(X22)

        D = self.D_cheb
        ik = 1j * k_c

        # (1, 20) interaction.
        u_term_1 = (v1 * (D @ u20)) + (u20 * (ik * u1)) + (v20 * (D @ u1))

        # (1*, 22) interaction.
        u1_c = np.conj(u1)
        v1_c = np.conj(v1)
        u_term_2 = (u1_c * (2 * ik * u22)) + (v1_c * (D @ u22)) + \
                   (u22 * (-ik * u1_c)) + (v22 * (D @ u1_c))

        N_u_3 = -(u_term_1 + u_term_2)

        zeros = np.zeros_like(u1)
        return np.concatenate([N_u_3, zeros, zeros, zeros])

    def solve_second_order(self, forcing, k_out, omega_shift):
        """Solve ``(A(k_out) - omega_shift B(k_out)) X = forcing``.

        Args:
            forcing: Second-order forcing vector.
            k_out: Output wavenumber.
            omega_shift: Frequency shift multiplying the mass matrix.

        Returns:
            Second-order response vector.
        """
        A, B = assemble_swe_matrices(
            self.D_cheb,
            self.y_cheb,
            alpha=k_out,
            beta=self.params.get("beta", 10.0),
            Cf=self.params.get("Cf", 0.01),
            Fr=self.params.get("Fr", 0.5),
            N_curv=0.0,
            sigma_width=0.0,
            params=self.params,
        )

        L = A - omega_shift * B

        M = self.N + 1
        forcing = np.array(forcing, copy=True)
        for r_idx in [0, M - 1]:
            forcing[1 * M + r_idx] = 0.0

        try:
            return la.solve(L, forcing)
        except la.LinAlgError:
            # Near-singular k = 0 resonance: regularize with a small shift.
            return la.solve(L + 1e-6 * np.eye(L.shape[0]), forcing)

    def compute_landau_coefficients(self, sigma_width, k_c):
        """Compute the single-mode cubic Landau coefficient.

        Args:
            sigma_width: Width gradient ``sigma``.
            k_c: Carrier wavenumber.

        Returns:
            Complex cubic coefficient ``gamma``.
        """
        omega = self.solve_linear_adjoint(sigma_width, k_c)
        N20, N22 = self.compute_nonlinear_forcing(k_c)

        X22 = self.solve_second_order(N22, 2 * k_c, 2 * omega)
        X20 = self.solve_second_order(N20, 0.0, 2 * np.real(omega))

        u1, v1, eta1, zb1 = self.extract_components(self.vector_linear)
        N3 = self.compute_cubic_interaction(u1, v1, eta1, zb1, X20, X22, k_c)

        return np.vdot(self.vector_adjoint, N3)
