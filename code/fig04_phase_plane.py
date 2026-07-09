"""
Figure 4: Phase Plane Trajectories in Amplitude Space
Shows evolution toward Mixed State from various initial conditions.
"""

import numpy as np
import matplotlib.pyplot as plt
import figure_utils
from scipy.integrate import solve_ivp


def main():
    figure_utils.set_jfm_style()

    # Coefficients from calculate_coupled_landau.py
    sigma_a = 0.3085 - 5.9616j
    sigma_b = 0.1629 - 4.7062j

    g_aa = -0.0030 + 0.0773j
    g_bb = 0.0548 + 1.0238j
    g_ab = 0.0137 + 0.4416j
    g_ba = -0.0129 + 0.1475j

    # Landau system (amplitude dynamics)
    # Landau system (amplitude dynamics) with 5th-order stabilizing term
    g5 = 1e-6
    def landau_system(t, y):
        A = y[0] + 1j*y[1]
        B = y[2] + 1j*y[3]

        dA = sigma_a*A - g_aa*(np.abs(A)**2)*A - g_ab*(np.abs(B)**2)*A - g5*(np.abs(A)**4)*A
        dB = sigma_b*B - g_bb*(np.abs(B)**2)*B - g_ba*(np.abs(A)**2)*B

        return [dA.real, dA.imag, dB.real, dB.imag]

    # Multiple initial conditions
    A_eq_pure = 0.0  # Subcritical, no stable pure state at this order
    colors = figure_utils.get_jfm_colors()

    initial_conditions = [
        ([1.0, 0.0, 1.0, 0.0], 'Small perturbation', colors[0]), # Orange
        ([25.0, 0.0, 2.0, 0.0], 'Central dominant', colors[4]), # Blue
        ([15.0, 0.0, 10.0, 0.0], 'Comparable amplitudes', colors[2]), # Bluish Green
        ([2.0, 0.0, 12.0, 0.0], 'Alternate dominant', colors[6]), # Reddish Purple
    ]

    t_span = (0, 15)
    t_eval = np.linspace(0, 15, 500)

    # Plot setup
    fig, ax = plt.subplots(figsize=(5, 4))

    for y0, label, color in initial_conditions:
        sol = solve_ivp(landau_system, t_span, y0, t_eval=t_eval, rtol=1e-8, atol=1e-10)

        A = sol.y[0] + 1j*sol.y[1]
        B = sol.y[2] + 1j*sol.y[3]

        amp_A = np.abs(A)
        amp_B = np.abs(B)

        ax.plot(amp_A, amp_B, color=color, linewidth=1.2, label=label)

        # Start marker
        ax.plot(amp_A[0], amp_B[0], 'o', color=color, markersize=5)
        # End marker
        ax.plot(amp_A[-1], amp_B[-1], 's', color=color, markersize=6)

    # Mark equilibrium points
    # Pure Central: (A_eq, 0)
    # Pure Central is subcritical, not plotted.
    # Final Mixed State (from simulation of stabilized system)
    mixed_A = 20.42
    mixed_B = 10.06
    ax.plot(mixed_A, mixed_B, 'k*', markersize=10, label='Mixed State eq.')

    # Labels
    ax.set_xlabel(r'Central bar amplitude $|A|$')
    ax.set_ylabel(r'Alternate bar amplitude $|B|$')
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 15)

    ax.legend(loc='upper left', fontsize=8, frameon=True)

    plt.tight_layout()

    # Save
    figure_utils.save_for_jfm(fig, 'Fig4_phase_plane')
    print("Figure 4 saved.")


if __name__ == "__main__":
    main()
