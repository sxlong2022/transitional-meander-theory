"""Coupled two-mode weakly nonlinear analysis for central and alternate bars.

Two bar modes coexist in a transitional reach: the symmetric central bar
(``m = 2``) at wavenumber ``k_A`` and the antisymmetric alternate bar
(``m = 1``) at wavenumber ``k_B``.  Writing their complex amplitudes as ``A(t)``
and ``B(t)``, the solvability condition at third order yields the coupled
Stuart--Landau system

.. math::

    \\frac{dA}{dt} = \\lambda_A A - \\gamma_{AA}|A|^2 A
                     - \\gamma_{AB}|B|^2 A,
    \\frac{dB}{dt} = \\lambda_B B - \\gamma_{BB}|B|^2 B
                     - \\gamma_{BA}|A|^2 B.

Each coefficient is an adjoint projection of the third-order forcing generated
by one nonlinear interaction.  Self-saturation (``gamma_AA``, ``gamma_BB``)
comes from the self mean-flow distortion and the self second harmonic.
Cross-coupling comes from the mean-flow distortion of one mode projected onto
the other, plus the mixed sum harmonic at ``k_A + k_B``.

The sign of ``Re(gamma_BA)`` carries the physical content: negative values mean
the central bar enhances alternate-bar growth, positive values mean competition.
"""

import os
import sys

import numpy as np
import scipy.linalg as la

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import linear_solver as swe
from landau_base import LandauCalculator


class LandauCoupledCalculator(LandauCalculator):
    """Adjoint projection machinery for the four coupled Landau coefficients."""

    def __init__(self, params, N, L_domain=None, sigma_width=0.1):
        """Initialize the calculator.

        Args:
            params: Physical parameter dictionary.
            N: Chebyshev polynomial degree.
            L_domain: Optional domain length, retained for compatibility.
            sigma_width: Width gradient ``sigma`` used in every matrix assembly.
        """
        super().__init__(params, N, L_domain)
        self.sigma_width = sigma_width

    def solve_adjoint(self, A, B, eigenmode_vec, omega=None):
        """Solve the adjoint eigenproblem for a given direct mode.

        Candidates are ranked by their projection onto the direct mode.  When
        ``omega`` is supplied the search is restricted to the neighbourhood of
        ``conj(omega)``, which keeps the adjoint paired with the same branch.

        Args:
            A: Direct operator matrix.
            B: Direct mass matrix.
            eigenmode_vec: Direct eigenvector.
            omega: Optional direct eigenvalue used for frequency matching.

        Returns:
            Adjoint eigenvector normalized so that ``<adj, B direct> = 1``.
        """
        vals_adj, vecs_adj = la.eig(A.conj().T, B.conj().T)

        valid = np.isfinite(vals_adj)
        vals_adj = vals_adj[valid]
        vecs_adj = vecs_adj[:, valid]
        if vals_adj.size == 0:
            raise ValueError("No finite eigenvalues found in the adjoint spectrum.")

        if omega is not None:
            target = np.conj(omega)
            dists = np.abs(vals_adj - target)
            close_mask = dists < (np.min(dists) + 0.1)
            candidate_vecs = (
                vecs_adj[:, close_mask] if np.any(close_mask) else vecs_adj
            )
            projections = np.abs(candidate_vecs.conj().T @ B @ eigenmode_vec)
            adj_vec = candidate_vecs[:, np.argmax(projections)]
        else:
            projections = np.abs(vecs_adj.conj().T @ B @ eigenmode_vec)
            adj_vec = vecs_adj[:, np.argmax(projections)]

        norm_factor = np.vdot(adj_vec, B @ eigenmode_vec)
        if abs(norm_factor) <= 1e-14:
            raise ValueError(
                "Adjoint normalization factor is near-zero "
                f"({norm_factor}); direct and adjoint modes are orthogonal."
            )
        return adj_vec / np.conj(norm_factor)

    def compute_nonlinear_forcing_harmonic_general(self, vec_a, vec_b, k_a, k_b):
        """Sum-harmonic forcing at ``k_a + k_b`` for two distinct modes.

        Args:
            vec_a: First state vector.
            vec_b: Second state vector.
            k_a: Wavenumber of ``vec_a``.
            k_b: Wavenumber of ``vec_b``.

        Returns:
            Forcing vector of length ``4(N + 1)`` with zero bed block.
        """
        u_a, v_a, eta_a, _ = self.extract_components(vec_a)
        u_b, v_b, eta_b, _ = self.extract_components(vec_b)

        term1_u = u_a * (1j * k_b * u_b) + v_a * (self.D_cheb @ u_b)
        term2_u = u_b * (1j * k_a * u_a) + v_b * (self.D_cheb @ u_a)
        forcing_u = -(term1_u + term2_u)

        term1_v = u_a * (1j * k_b * v_b) + v_a * (self.D_cheb @ v_b)
        term2_v = u_b * (1j * k_a * v_a) + v_b * (self.D_cheb @ v_a)
        forcing_v = -(term1_v + term2_v)

        flux_x = u_a * eta_b + u_b * eta_a
        flux_y = v_a * eta_b + v_b * eta_a
        k_sum = k_a + k_b
        forcing_h = -(1j * k_sum * flux_x + self.D_cheb @ flux_y)

        M = self.N + 1
        F = np.zeros(4 * M, dtype=complex)
        F[0:M] = forcing_u
        F[M:2 * M] = forcing_v
        F[2 * M:3 * M] = forcing_h
        return F

    def compute_full_coefficients(self, vec_a, sigma_a, vec_b, sigma_b,
                                  k_a, k_b=None, omega_sum=None):
        """Compute the four coupled Landau coefficients.

        Two assembly modes are supported:

        * **Dual wavenumber** (``k_b`` given): independent direct and adjoint
          operators at ``k_A`` and ``k_B``, with the mixed sum harmonic at
          ``k_A + k_B``.
        * **Single carrier** (``k_b`` omitted): sets ``k_b = k_a``, assembles both
          direct operators on the ``k_a`` grid with the second harmonic at ``2 k_a``,
          and selects ``adj_a`` by direct projection (``omega=None``) rather than
          frequency matching.  This is the illustrative baseline of main-text Table 2.

        Args:
            vec_a: Central-bar direct eigenvector.
            sigma_a: Central-bar eigenvalue.
            vec_b: Alternate-bar direct eigenvector.
            sigma_b: Alternate-bar eigenvalue.
            k_a: Wavenumber of mode A.
            k_b: Optional wavenumber of mode B; defaults to ``k_a``.
            omega_sum: Optional frequency shift for the mixed sum harmonic.

        Returns:
            ``(gamma_aa, gamma_ab, gamma_ba, gamma_bb)`` as complex numbers.
        """
        is_single_carrier = k_b is None
        if k_b is None:
            k_b = k_a

        print(f"Solving adjoints for k_A={k_a}, k_B={k_b} ...")

        A_lin_a, B_lin_a = swe.assemble_swe_matrices(
            self.D_cheb, self.y_cheb, k_a,
            self.params["beta"], self.params["Cf"], self.params["Fr"],
            N_curv=0.0, sigma_width=self.sigma_width, params=self.params,
        )
        A_lin_b, B_lin_b = swe.assemble_swe_matrices(
            self.D_cheb, self.y_cheb, k_b,
            self.params["beta"], self.params["Cf"], self.params["Fr"],
            N_curv=0.0, sigma_width=self.sigma_width, params=self.params,
        )

        # In single-carrier mode the mode-A adjoint projects on the carrier grid
        # without frequency matching, consistent with the baseline assembly.
        adj_a = self.solve_adjoint(
            A_lin_a, B_lin_a, vec_a,
            omega=None if is_single_carrier else sigma_a,
        )
        adj_b = self.solve_adjoint(A_lin_b, B_lin_b, vec_b, omega=sigma_b)

        # 1. Self coefficients.
        print("Computing self interactions ...")
        gamma_aa = self.calculate_single_gamma(vec_a, sigma_a, adj_a, k_a)
        gamma_bb = self.calculate_single_gamma(vec_b, sigma_b, adj_b, k_b)

        # 2. Cross coefficients.
        print("Computing cross interactions ...")
        psi_0_bb = self.solve_mean_flow(vec_b, k_b)
        psi_0_aa = self.solve_mean_flow(vec_a, k_a)
        psi_2_ab = self.solve_harmonic_sum(vec_a, vec_b, k_a, k_b,
                                           omega_sum=omega_sum)
        k_sum = k_a + k_b

        # gamma_AB: effect of |B|^2 on A, projected onto k_A.
        cubic_ab_1 = self.compute_cubic_interaction_mixed(vec_a, psi_0_bb, k_a, 0)
        cubic_ab_2 = self.compute_cubic_interaction_mixed(
            vec_b.conj(), psi_2_ab, -k_b, k_sum
        )
        gamma_ab = -np.vdot(adj_a, cubic_ab_1 + cubic_ab_2)

        # gamma_BA: effect of |A|^2 on B, projected onto k_B.
        cubic_ba_1 = self.compute_cubic_interaction_mixed(vec_b, psi_0_aa, k_b, 0)
        cubic_ba_2 = self.compute_cubic_interaction_mixed(
            vec_a.conj(), psi_2_ab, -k_a, k_sum
        )
        gamma_ba = -np.vdot(adj_b, cubic_ba_1 + cubic_ba_2)

        return gamma_aa, gamma_ab, gamma_ba, gamma_bb

    def solve_mean_flow(self, vec, k):
        """Solve the zero-wavenumber mean-flow response.

        The ``k = 0`` operator is singular, so the solve is a truncated
        least-squares projection.

        Args:
            vec: Mode state vector driving the Reynolds stresses.
            k: Wavenumber of ``vec``.

        Returns:
            Mean-flow response vector.
        """
        forcing = self.compute_nonlinear_forcing_mean_isolated(vec, vec, k)
        A0, _ = swe.assemble_swe_matrices(
            self.D_cheb, self.y_cheb, 0.0,
            self.params["beta"], self.params["Cf"], self.params["Fr"],
            N_curv=0.0, sigma_width=self.sigma_width, params=self.params,
        )
        X, _, _, _ = la.lstsq(A0, forcing, cond=1e-10)
        return X

    def solve_harmonic_sum(self, vec1, vec2, k1, k2=None, omega_sum=None):
        """Solve the second-order response at ``k1 + k2``.

        Args:
            vec1: First state vector.
            vec2: Second state vector.
            k1: Wavenumber of ``vec1``.
            k2: Optional wavenumber of ``vec2``; defaults to ``k1``.
            omega_sum: Optional frequency shift applied to the mass matrix.

        Returns:
            Second-order response vector.
        """
        if k2 is None:
            k2 = k1
        k_sum = k1 + k2
        forcing = self.compute_nonlinear_forcing_harmonic_general(vec1, vec2, k1, k2)
        A2, B2 = swe.assemble_swe_matrices(
            self.D_cheb, self.y_cheb, k_sum,
            self.params["beta"], self.params["Cf"], self.params["Fr"],
            N_curv=0.0, sigma_width=self.sigma_width, params=self.params,
        )

        if omega_sum is not None:
            L2 = A2 - omega_sum * B2
            try:
                return la.solve(L2, forcing)
            except la.LinAlgError:
                X, _, _, _ = la.lstsq(L2, forcing, cond=1e-10)
                return X

        try:
            return la.solve(A2, forcing)
        except la.LinAlgError:
            X, _, _, _ = la.lstsq(A2, forcing, cond=1e-10)
            return X

    def calculate_single_gamma(self, vec, sigma, adj, k):
        """Project the self-interaction forcing of one mode onto its adjoint.

        Args:
            vec: Mode state vector.
            sigma: Mode eigenvalue.
            adj: Adjoint state vector.
            k: Mode wavenumber.

        Returns:
            Complex self-saturation coefficient.
        """
        psi0 = self.solve_mean_flow(vec, k)
        psi2 = self.solve_harmonic_sum(vec, vec, k, k)
        force0 = self.compute_cubic_interaction_mixed(vec, psi0, k, 0)
        force2 = self.compute_cubic_interaction_mixed(vec.conj(), psi2, -k, 2 * k)
        return -np.vdot(adj, force0 + force2)
