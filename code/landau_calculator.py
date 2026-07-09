import numpy as np
import scipy.linalg as la
import linear_solver as swe
from landau_base import LandauCalculator


class LandauCoupledCalculator(LandauCalculator):
    def __init__(self, params, N, L_domain=None, sigma_width=0.1):
        super().__init__(params, N, L_domain)
        self.sigma_width = sigma_width  # Store for use in matrix assemblies

    def solve_adjoint(self, A, B, eigenmode_vec):
        """
        Solves for the adjoint mode corresponding to the given direct eigenmode.
        Normalizes such that <adj, B * direct> = 1.
        """
        # Adjoint problem: A^H Y = conj(omega) B^H Y
        vals_adj, vecs_adj = la.eig(A.conj().T, B.conj().T)

        # Heuristic: Maximize projection
        projections = np.abs(vecs_adj.conj().T @ B @ eigenmode_vec)
        idx = np.argmax(projections)

        adj_vec = vecs_adj[:, idx]

        # Normalize
        norm_factor = np.vdot(adj_vec, B @ eigenmode_vec)
        adj_vec = adj_vec / np.conj(norm_factor)

        return adj_vec

    def compute_full_coefficients(self, vec_a, sigma_a, vec_b, sigma_b, k):
        """
        Computes gamma_aa, gamma_ab, gamma_ba, gamma_bb
        """
        print("Solving Adjoints...")
        # Get system matrices
        A_lin, B_lin = swe.assemble_swe_matrices(self.D_cheb, self.y_cheb, k,
                                                self.params['beta'], self.params['Cf'], self.params['Fr'],
                                                N_curv=0.0, sigma_width=self.sigma_width, params=self.params)

        adj_a = self.solve_adjoint(A_lin, B_lin, vec_a)
        adj_b = self.solve_adjoint(A_lin, B_lin, vec_b)

        # 1. Self Coefficients (Gamma_AA, Gamma_BB)
        print("Computing Self Interactions...")
        gamma_aa = self.calculate_single_gamma(vec_a, sigma_a, adj_a, k)
        gamma_bb = self.calculate_single_gamma(vec_b, sigma_b, adj_b, k)

        # 2. Cross Interactions
        print("Computing Cross Interactions...")

        # Step 1: Calculate the relevant 2nd order modes
        psi_0_bb = self.solve_mean_flow(vec_b, k)
        psi_0_aa = self.solve_mean_flow(vec_a, k)

        psi_2_ab = self.solve_harmonic_sum(vec_a, vec_b, k)

        # Step 2: Calculate Cubic Terms and Project
        # Gamma_AB: Effect of |B|^2 on A.
        cubic_1 = self.compute_cubic_interaction_mixed(vec_a, psi_0_bb, k, 0)
        cubic_2 = self.compute_cubic_interaction_mixed(vec_b.conj(), psi_2_ab, -k, 2*k)

        total_force_ab = cubic_1 + cubic_2
        proj_ab = np.vdot(adj_a, total_force_ab)
        gamma_ab = -proj_ab

        # --- Gamma_BA (Force on B from A) ---
        cubic_ba_1 = self.compute_cubic_interaction_mixed(vec_b, psi_0_aa, k, 0)
        cubic_ba_2 = self.compute_cubic_interaction_mixed(vec_a.conj(), psi_2_ab, -k, 2*k)

        total_force_ba = cubic_ba_1 + cubic_ba_2
        proj_ba = np.vdot(adj_b, total_force_ba)
        gamma_ba = -proj_ba

        return gamma_aa, gamma_ab, gamma_ba, gamma_bb

    def solve_mean_flow(self, vec, k):
        forcing = self.compute_nonlinear_forcing_mean_isolated(vec, vec, k)
        A0, B0 = swe.assemble_swe_matrices(self.D_cheb, self.y_cheb, 0.0,
                                          self.params['beta'], self.params['Cf'], self.params['Fr'],
                                          N_curv=0.0, sigma_width=self.sigma_width, params=self.params)
        # Use least-squares solve for numerical stability (k=0 matrix is near-singular)
        X, residuals, rank, s = la.lstsq(A0, forcing, cond=1e-10)
        return X

    def solve_harmonic_sum(self, vec1, vec2, k):
        forcing = self.compute_nonlinear_forcing_harmonic_mixed(vec1, vec2, k)
        A2, B2 = swe.assemble_swe_matrices(self.D_cheb, self.y_cheb, 2*k,
                                          self.params['beta'], self.params['Cf'], self.params['Fr'],
                                          N_curv=0.0, sigma_width=self.sigma_width, params=self.params)
        X = la.solve(A2, forcing)
        return X

    def calculate_single_gamma(self, vec, sigma, adj, k):
        psi0 = self.solve_mean_flow(vec, k)
        psi2 = self.solve_harmonic_sum(vec, vec, k)
        force0 = self.compute_cubic_interaction_mixed(vec, psi0, k, 0)
        force2 = self.compute_cubic_interaction_mixed(vec.conj(), psi2, -k, 2*k)
        total = force0 + force2
        gamma = -np.vdot(adj, total)
        return gamma
