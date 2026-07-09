
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to access 'theory' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory.os import morphodynamic as swe
import scipy.linalg as la
import figure_utils as fu

def get_cheb_grid(N):
    """Return Chebyshev nodes and Diff matrix."""
    # Simple Chebyshev impl or import if available
    # For now, let's use a quick local impl or import from landau if possible
    # Actually, theory.nonlinear.landau has a LandauCalculator that does this.
    # Let's import LandauCalculator to use its grid setup, or just copy-paste grid logic.
    t = np.linspace(0, np.pi, N+1)
    y = np.cos(t)
    
    # Differentiation Matrix D (Trefethen)
    c = np.hstack(([2], np.ones(N-1), [2])) * (-1)**np.arange(N+1)
    X = np.tile(y, (N+1, 1))
    dX = X - X.T
    D = (c[:, None] / c[None, :]) / (dX + np.eye(N+1))
    D = D - np.diag(D.sum(axis=1))
    return y, D

def solve_mode(N, beta, Cf, Fr, alpha, sigma):
    y, D = get_cheb_grid(N)
    params = {'beta': beta, 'Cf': Cf, 'Fr': Fr, 'transport_exponent_u': 3.0}
    A, B = swe.assemble_swe_matrices(D, y, alpha, beta, Cf, Fr, N_curv=0.0, sigma_width=sigma, params=params)
    
    vals, vecs = la.eig(A, B)
    
    # Filter Infinite eigenvalues (arising from algebraic constraints B*v=0)
    finite_mask = np.isfinite(vals)
    vals = vals[finite_mask]
    vecs = vecs[:, finite_mask]
    
    idx = np.argsort(vals.real)[::-1] # Sort desc
    vals = vals[idx]
    vecs = vecs[:, idx]
    
    # Extract vars
    M = N + 1
    # Check 1st mode
    vec = vecs[:, 0]
    u = vec[0:M]
    v = vec[M:2*M]
    eta = vec[2*M:3*M]
    zb = vec[3*M:4*M]
    return vals[0], u, v, eta, zb

from scipy.interpolate import BarycentricInterpolator

def main():
    # Set publication style
    fu.set_publication_style()
    
    # Parameters matches report
    sigma = 0.15
    beta = 15.0
    Cf = 0.01
    Fr = 0.5
    
    # Computed Amplitudes for Beta=15, Sigma=0.15
    amp_A = 20.42   # Central
    amp_B = 10.06   # Alternate
    
    k_A = 2.0
    k_B = 1.0
    
    print("Calculating Mode Shapes...")
    # Solve on Chebyshev Grid (N=31)
    omega_A, u_A, v_A, eta_A, zb_A = solve_mode(31, beta, Cf, Fr, k_A, sigma)
    omega_B, u_B, v_B, eta_B, zb_B = solve_mode(31, beta, Cf, Fr, k_B, sigma)

    # Grid for Plotting (Upsampled)
    L_plot = 20.0
    Nx = 600 # High res streamwise
    Ny_fine = 101 # High res transverse
    
    x = np.linspace(0, L_plot, Nx)
    # y comes from Chebyshev grid (N=31)
    y_cheb, _ = get_cheb_grid(31) 
    
    # Create Fine Grid
    y_fine = np.linspace(-1, 1, Ny_fine)
    X, Y = np.meshgrid(x, y_fine)
    
    # Interpolate Modes to Fine Grid using Barycentric (spectral accuracy)
    interp_A = BarycentricInterpolator(y_cheb, zb_A)
    interp_B = BarycentricInterpolator(y_cheb, zb_B)
    
    zb_A_fine = interp_A(y_fine)
    zb_B_fine = interp_B(y_fine)
    
    # Normalize (Optional: but ensures amplitudes make sense relative to max=1 mode)
    zb_A_fine = zb_A_fine / np.max(np.abs(zb_A_fine))
    zb_B_fine = zb_B_fine / np.max(np.abs(zb_B_fine))

    # Reconstruct Field on Fine Grid
    # Mode A (Central)
    Z_A = (amp_A * zb_A_fine[:, None] * np.exp(1j * k_A * X)).real
    
    # Mode B (Alternate)
    Z_B = (amp_B * zb_B_fine[:, None] * np.exp(1j * k_B * X)).real
    
    # Total
    Z_total = Z_A + Z_B
    
    # Plot
    fig, ax = plt.subplots(figsize=(6.5, 2.5)) 
    
    # Diverging colormap - ENHANCED CONTRAST with FIXED RANGE
    # Use a fixed symmetric range for clearer visualization
    vmax = 0.3  # Fixed value to saturate colors earlier
    levels = np.linspace(-vmax, vmax, 31)
    
    im = ax.contourf(X, Y, Z_total, levels=levels, cmap='RdBu_r', extend='both')
    
    # Decorations - Use curvilinear coordinates (s, n) for meandering rivers
    ax.set_xlabel('Streamwise coordinate $s$')
    ax.set_ylabel('Transverse coordinate $n$')
    ax.set_yticks([-1, 0, 1])
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Bed elevation $z_b$')
    # Regular tick values
    cbar.set_ticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
    
    # No title for publication
    
    # Save
    # figure_utils.save_for_publication now handles both PDF and PNG generation relative to base name
    fu.save_for_publication(fig, 'Fig5_mixed_state')

if __name__ == "__main__":
    main()
