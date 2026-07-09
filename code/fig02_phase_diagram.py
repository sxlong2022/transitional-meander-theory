"""
Figure 2: Linear Stability Phase Diagram
Shows growth rates of Central Bar vs Alternate Bar across sigma range.
"""

import numpy as np
import sys
import os

import matplotlib.pyplot as plt
import figure_utils
import linear_solver as swe
import diagnostics as diag
import scipy.linalg as la

def cheb_collocation(N):
    """Chebyshev differentiation matrix."""
    x = np.cos(np.pi * np.arange(N + 1) / N)
    c = np.hstack(([2], np.ones(N - 1), [2])) * (-1)**np.arange(N + 1)
    X = np.tile(x, (N + 1, 1))
    dX = X - X.T
    D = (c[:, np.newaxis] / c[np.newaxis, :]) / (dX + np.eye(N + 1))
    D = D - np.diag(np.sum(D, axis=1))
    return D, x

def main():
    figure_utils.set_publication_style()
    
    # Parameters
    params = {
        'beta': 10.0,
        'Cf': 0.01,
        'Fr': 0.5,
        'transport_exponent_u': 3.0
    }
    N = 16
    
    # Sweep range
    sigmas = np.linspace(-0.1, 0.2, 31)
    test_ks = [0.4, 0.5, 0.6, 2.0, 2.5, 3.0]
    
    D, y = cheb_collocation(N)
    
    results = {
        'sigma': [],
        'central': [],
        'alternate': []
    }
    
    print("Sweeping sigma for phase diagram...")
    
    for s_val in sigmas:
        best_central = -999.0
        best_alternate = -999.0
        
        for k_test in test_ks:
            A_mat, B_mat = swe.assemble_swe_matrices(
                D, y, k_test, 
                params['beta'], params['Cf'], params['Fr'], 
                N_curv=0.0, sigma_width=s_val, params=params
            )
            
            vals, vecs = la.eig(A_mat, B_mat)
            
            valid = np.isfinite(vals) & (np.real(vals) < 10.0)
            vals = vals[valid]
            vecs = vecs[:, valid]
            
            idx = np.argsort(np.real(vals))[::-1]
            M = N + 1
            
            for i in range(min(10, len(idx))):
                mode_id = idx[i]
                gr = np.real(vals[mode_id])
                vec = vecs[:, mode_id]
                zb = vec[3*M:4*M]
                
                sym = diag.calculate_symmetry_index(zb, N)
                
                if sym < 0.2 and gr > best_central:
                    best_central = gr
                elif sym > 0.5 and gr > best_alternate:
                    best_alternate = gr
        
        if best_central < -100: best_central = np.nan
        if best_alternate < -100: best_alternate = np.nan
        
        results['sigma'].append(s_val)
        results['central'].append(best_central)
        results['alternate'].append(best_alternate)
    
    # Plotting
    colors = figure_utils.get_publication_colors()
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    
    ax.plot(results['sigma'], results['central'], marker='o', color=colors[4], # Blue
            markersize=4, linewidth=1.2, label='Central bar (symmetric)')
    ax.plot(results['sigma'], results['alternate'], marker='s', color=colors[6], # Reddish Purple
            markersize=4, linewidth=1.2, label='Alternate bar (antisymmetric)')
    
    ax.axhline(0, color='gray', linestyle=':', linewidth=0.8)
    ax.axvline(0, color='gray', linestyle=':', linewidth=0.8)
    
    # Find and mark crossover point
    central_arr = np.array(results['central'])
    alternate_arr = np.array(results['alternate'])
    sigma_arr = np.array(results['sigma'])
    
    # Find approximate crossover (where curves intersect)
    diff = central_arr - alternate_arr
    # Handle NaN values
    valid_mask = np.isfinite(diff)
    if np.any(valid_mask):
        sign_changes = np.where(np.diff(np.sign(diff[valid_mask])))[0]
        if len(sign_changes) > 0:
            cross_idx = sign_changes[0]
            sigma_cross = sigma_arr[valid_mask][cross_idx]
            ax.axvline(sigma_cross, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)
            ax.annotate(f'$\\sigma_c \\approx {sigma_cross:.2f}$', 
                       xy=(sigma_cross, 2.0), fontsize=8, ha='left')
    
    ax.set_xlabel(r'Width gradient $\sigma$')
    ax.set_ylabel(r'Growth rate $\mathrm{Re}(\omega)$')
    ax.set_xlim(-0.1, 0.2)
    ax.set_ylim(-0.5, 2.5)
    
    # Simplified legend (only 2 entries now)
    ax.legend(loc='upper right', fontsize=8, frameon=True)
    
    plt.tight_layout()
    
    figure_utils.save_for_publication(fig, 'Fig2_phase_diagram')
    print("Figure 2 saved.")

if __name__ == "__main__":
    main()
