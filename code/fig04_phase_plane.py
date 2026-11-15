"""Figure 4: phase-plane trajectories of the coupled amplitude moduli.

The complex Stuart--Landau system reduces, for the moduli ``R_A = |A|`` and
``R_B = |B|``, to

.. math::

    \\frac{dR_A}{dt} = \\sigma_A R_A - \\mathrm{Re}(\\gamma_{AA}) R_A^3
                      - \\mathrm{Re}(\\gamma_{AB}) R_B^2 R_A
                      - g_5 R_A^5,
    \\frac{dR_B}{dt} = \\sigma_B R_B - \\mathrm{Re}(\\gamma_{BB}) R_B^3
                      - \\mathrm{Re}(\\gamma_{BA}) R_A^2 R_B.

The fifth-order term ``-g_5 |A|^4 A`` with ``g_5 = 1e-6`` bounds the subcritical
central-bar branch, which has ``Re(gamma_AA) < 0`` and would otherwise grow
without limit.  Trajectories from four initial conditions converge on the stable
mixed-state node.  Both nullclines and the isolated alternate-bar equilibrium
``|B_iso| = sqrt(sigma_B / Re(gamma_BB))`` are marked for reference.

Run from the repository root::

    python code/fig04_phase_plane.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

import figure_utils


def mixed_state_amplitudes(sigma_a, sigma_b, g_aa, g_bb, g_ab, g_ba, g5):
    """Return the positive mixed-state moduli ``(|A_eq|, |B_eq|)``.

    Setting the time derivatives to zero and writing ``x = |A|^2``,
    ``y = |B|^2`` gives ``y = (sigma_b - g_ba x) / g_bb`` and a quadratic
    ``g5 x^2 - c1 x - c0 = 0`` with

    ``c0 = sigma_a - g_ab sigma_b / g_bb`` and
    ``c1 = -(g_aa - g_ab g_ba / g_bb)``.

    Args:
        sigma_a, sigma_b: Real linear growth rates.
        g_aa, g_bb, g_ab, g_ba: Real parts of the Landau coefficients.
        g5: Fifth-order regularization coefficient.

    Returns:
        ``(r_a, r_b)``, the positive equilibrium moduli.
    """
    c0 = sigma_a - (g_ab * sigma_b) / g_bb
    c1 = -(g_aa - (g_ab * g_ba) / g_bb)
    disc = c1**2 + 4.0 * g5 * c0
    x = (c1 + np.sqrt(disc)) / (2.0 * g5)
    y = (sigma_b - g_ba * x) / g_bb
    return np.sqrt(x), np.sqrt(y)


def main():
    figure_utils.set_publication_style()
    colors = figure_utils.get_publication_colors()

    # Real linear growth rates: sigma_A = Re(lambda_A), sigma_B = Re(lambda_B).
    sigma_a = 0.3085
    sigma_b = 0.1629

    # Real parts of the Landau coefficients at beta = 15.0, sigma = 0.15.
    g_aa = -0.0030
    g_bb = 0.0548
    g_ab = 0.0137
    g_ba = -0.0129

    g5 = 1e-6

    def landau_modulus_system(t, y):
        """Modulus dynamics decoupled from the fast carrier oscillation."""
        RA, RB = y[0], y[1]
        dRA = sigma_a * RA - g_aa * (RA**3) - g_ab * (RB**2) * RA - g5 * (RA**5)
        dRB = sigma_b * RB - g_bb * (RB**3) - g_ba * (RA**2) * RB
        return [dRA, dRB]

    initial_conditions = [
        ([1.0, 1.0], "Small perturbation", colors[0]),
        ([25.0, 2.0], "Central dominant", colors[4]),
        ([15.0, 10.0], "Comparable amplitudes", colors[2]),
        ([2.0, 12.0], "Alternate dominant", colors[6]),
    ]

    t_span = (0, 30)
    t_eval = np.linspace(0, 30, 600)

    fig, ax = plt.subplots(figsize=(5.5, 4.2))

    # R_B nullcline: sigma_B - g_bb R_B^2 - g_ba R_A^2 = 0.
    ra_null = np.linspace(0, 28, 300)
    rb_null = np.sqrt(np.maximum(0, (sigma_b - g_ba * (ra_null**2)) / g_bb))
    ax.plot(ra_null, rb_null, "--", color="gray", alpha=0.7, linewidth=1.0,
            label=r"$|B|$-nullcline ($\mathrm{d}|B|/\mathrm{d}t=0$)")

    for y0, label, color in initial_conditions:
        sol = solve_ivp(landau_modulus_system, t_span, y0, t_eval=t_eval,
                        rtol=1e-9, atol=1e-11)
        amp_A = sol.y[0]
        amp_B = sol.y[1]

        ax.plot(amp_A, amp_B, color=color, linewidth=1.3, label=label)
        ax.plot(amp_A[0], amp_B[0], "o", color=color, markersize=5)
        ax.plot(amp_A[-1], amp_B[-1], "s", color=color, markersize=5)

    # Isolated alternate-bar equilibrium, (0, B_iso).
    B_iso = np.sqrt(sigma_b / g_bb)
    ax.plot(0, B_iso, "d", color="dimgray", markersize=7,
            label=rf"Isolated $|B_{{\mathrm{{iso}}}}| = {B_iso:.2f}$")

    # Mixed-state node, evaluated from the same parameters that drive the flow.
    mixed_A, mixed_B = mixed_state_amplitudes(
        sigma_a, sigma_b, g_aa, g_bb, g_ab, g_ba, g5
    )
    amplification = mixed_B / B_iso
    ax.plot(mixed_A, mixed_B, "k*", markersize=11,
            label=r"Mixed state ($|A_{\mathrm{eq}}|, |B_{\mathrm{eq}}|$)")

    ax.annotate(
        rf"$\mathbf{{{amplification:.1f}\times}}$ amplification" + "\n"
        + rf"($|B_{{\mathrm{{eq}}}}| / |B_{{\mathrm{{iso}}}}| = {amplification:.2f}$)",
        xy=(mixed_A, mixed_B),
        xytext=(mixed_A - 8.5, mixed_B + 1.8),
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5),
    )

    ax.set_xlabel(r"Central bar amplitude $|A|$")
    ax.set_ylabel(r"Alternate bar amplitude $|B|$")
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 15)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=7.5, frameon=True, framealpha=0.9)

    plt.tight_layout()
    figure_utils.save_publication_figure(fig, "Fig4_phase_plane")

    print(f"Mixed state: |A_eq| = {mixed_A:.2f}, |B_eq| = {mixed_B:.2f}, "
          f"|B_iso| = {B_iso:.2f}, amplification = {amplification:.2f}x")
    print("Figure 4 written.")


if __name__ == "__main__":
    main()
