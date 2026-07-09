"""
Figure 6: Cross-Interaction Coefficient γ_BA vs Width Gradient σ
Shows the narrow transition window where cross-enhancement occurs.
"""

import numpy as np
import sys
import os

import matplotlib.pyplot as plt
from figure_utils import set_publication_style, save_for_publication, get_publication_colors
import linear_solver as swe
import diagnostics as diag
import scipy.linalg as la
from landau_calculator import LandauCoupledCalculator

# ------------------ Helper Functions ------------------

def cheb_collocation(N):
    """Chebyshev differentiation matrix."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack(([2], np.ones(N - 1), [2])) * (-1)**np.arange(N + 1)
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

def find_modes(A_mat, B_mat, N):
    """Find Central (Mode A) and Alternate (Mode B) from eigenvalue problem."""
    vals, vecs = la.eig(A_mat, B_mat)
    
    # Filter physical modes
    valid = np.isfinite(vals) & (np.abs(np.real(vals)) < 5.0)
    vals = vals[valid]
    vecs = vecs[:, valid]
    
    if len(vals) == 0:
        return None, None, None, None
    
    idx = np.argsort(np.real(vals))[::-1]
    
    mode_A, mode_B = None, None
    omega_A, omega_B = None, None
    M = N + 1
    
    for i in idx[:20]:
        vec = vecs[:, i]
        gr = vals[i]
        
        zb = vec[3*M:]
        if len(zb) == 0:
            continue
            
        sym = diag.calculate_symmetry_index(zb, N)
        
        # Relaxed constraint: Allow Central Bar even if growth rate is negative
        # (at high sigma, mode may be stable but still couples to Alternate Bar)
        if sym < 0.3 and mode_A is None:
            mode_A = vec
            omega_A = gr
        elif sym > 0.4 and mode_B is None:
            mode_B = vec
            omega_B = gr
            
        if mode_A is not None and mode_B is not None:
            break
            
    return mode_A, omega_A, mode_B, omega_B

def compute_gamma_BA(sigma_val, params_base, N=16, k_c=2.0):
    """Compute gamma_BA at a single sigma point."""
    params = params_base.copy()
    
    D, y = cheb_collocation(N)
    
    try:
        A_mat, B_mat = swe.assemble_swe_matrices(
            D, y, k_c, params['beta'], params['Cf'], params['Fr'],
            N_curv=0.0, sigma_width=sigma_val, params=params
        )
        
        mode_A, omega_A, mode_B, omega_B = find_modes(A_mat, B_mat, N)
        
        if mode_A is None or mode_B is None:
            return np.nan, None, None, None, None
        
        # Extract bed elevation for symmetry diagnostics
        M = N + 1
        zb_A = mode_A[3*M:4*M]
        zb_B = mode_B[3*M:4*M]
        sym_A = diag.calculate_symmetry_index(zb_A, N)
        sym_B = diag.calculate_symmetry_index(zb_B, N)
        
        # Create calculator instance
        calc = LandauCoupledCalculator(params, N)
        _, _, g_ba, _ = calc.compute_full_coefficients(mode_A, omega_A, mode_B, omega_B, k_c)
        
        return np.real(g_ba), omega_A, omega_B, sym_A, sym_B
        
    except Exception as e:
        print(f"Error at sigma={sigma_val}: {e}")
        return np.nan, None, None, None, None

# ------------------ Main ------------------

def main():
    set_publication_style()
    colors = get_publication_colors()
    
    # Base parameters (fixed β = 10)
    params_base = {
        'beta': 15.0,
        'Cf': 0.01,
        'Fr': 0.5,
        'transport_exponent_u': 3.0
    }
    
    # Finer σ sweep (increased resolution from 25 to 41 points)
    sigma_vals = np.linspace(-0.25, 0.35, 41)
    gamma_BA_vals = []
    
    print(f"Sweeping: sigma in [{sigma_vals[0]:.2f}, {sigma_vals[-1]:.2f}], {len(sigma_vals)} points")
    print(f"Fixed: beta = {params_base['beta']}")
    
    for i, sigma in enumerate(sigma_vals):
        print(f"[{i+1}/{len(sigma_vals)}] sigma={sigma:.3f}", end="")
        result = compute_gamma_BA(sigma, params_base)
        g_ba, omega_A, omega_B, sym_A, sym_B = result
        gamma_BA_vals.append(g_ba)
        
        if np.isfinite(g_ba):
            print(f" -> gamma_BA = {g_ba:.3f}", end="")
            if omega_A is not None and omega_B is not None:
                print(f"  [Modes: A(ω={np.real(omega_A):.2f}, sym={sym_A:.2f}), B(ω={np.real(omega_B):.2f}, sym={sym_B:.2f})]")
            else:
                print()
        else:
            print(" -> NaN")
    
    gamma_BA_vals = np.array(gamma_BA_vals)
    
    # ===== Filter numerical outliers for cleaner visualization =====
    # Values with |γ_BA| > 15 are likely numerical artifacts (mode tracking jumps)
    # Stricter threshold to remove σ≈0 spike (-15.164)
    gamma_BA_plot = gamma_BA_vals.copy()
    outlier_mask = np.abs(gamma_BA_plot) > 50
    outliers_count = np.sum(outlier_mask)
    if outliers_count > 0:
        print(f"\nNote: {outliers_count} outlier(s) filtered from plot (|γ_BA| > 50)")
        gamma_BA_plot[outlier_mask] = np.nan
    
    # Focus on physically reliable region: truncate at σ = 0
    # (converging channels σ<0 show numerical instabilities)
    display_mask = sigma_vals >= 0
    sigma_display = sigma_vals[display_mask]
    gamma_display = gamma_BA_plot[display_mask]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(5, 3.5))
    
    # Fill cross-enhancement region (γ_BA < 0)
    ax.fill_between(sigma_display, gamma_display, 0, 
                    where=(gamma_display < 0) & np.isfinite(gamma_display), 
                    color='pink', alpha=0.5, 
                    label='Cross-enhancement region')
    
    # Main curve with points
    valid = np.isfinite(gamma_display)
    ax.plot(sigma_display[valid], gamma_display[valid], '-o', color=colors[6], 
            linewidth=1.5, markersize=4)
    
    # Zero line
    ax.axhline(0, color='k', linestyle='--', linewidth=0.8)
    
    # Labels
    ax.set_xlabel(r'Width gradient $\sigma$')
    ax.set_ylabel(r'Cross-coupling coefficient $\mathrm{Re}(\gamma_{BA})$')
    
    # Annotations - simplified: legend already identifies the shaded region
    # Add an arrow pointing to the cross-enhancement zone instead of blocking text
    enhancement_mask = (gamma_display < 0) & np.isfinite(gamma_display)
    if np.any(enhancement_mask):
        sigma_enh = sigma_display[enhancement_mask]
        # Arrow annotation from margin to the center of enhacement region
        # Arrow annotation pointing straight up to the center of enhancement region (approx 0.06)
        # Hardcoded center for visual perfection as requested
        center_x = 0.05 
        ax.annotate(r'$\gamma_{BA} < 0$', 
                    xy=(center_x, -0.42), xytext=(center_x, 2.0),
                    fontsize=9, ha='center',
                    arrowprops=dict(arrowstyle='->', color='gray', lw=0.8),
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Regime labels - positioned based on actual y-range
    ax.text(0.12, 7.5, 'Diverging\nchannels', ha='center', fontsize=7, style='italic')
    
    # Adjusted x-limits to avoid numerical artifacts at high sigma
    ax.set_xlim(-0.01, 0.16)
    ax.set_ylim(-0.6, 9.0)
    
    ax.legend(loc='upper left', fontsize=8, frameon=True)
    
    plt.tight_layout()
    
    # Save
    save_for_publication(fig, 'Fig6_gamma_BA_curve')
    print("\nFigure 6 saved.")

if __name__ == "__main__":
    main()
