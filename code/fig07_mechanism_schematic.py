"""Figure 7: schematic of the nonlinear cross-enhancement mechanism.

The diagram traces the causal chain behind ``Re(gamma_BA) < 0``:

1. the finite-amplitude central bar (mode A) drives a zero-wavenumber mean-flow
   distortion ``delta u_0``;
2. that distortion modifies the effective stability of the alternate bar
   (mode B), producing cross-enhancement;
3. mode B feeds back on mode A through competitive suppression, closing the loop.

Inset panels show the transverse bed elevation of each mode to fix the
correspondence with Figures 3 and 5.

Run from the repository root::

    python code/fig07_mechanism_schematic.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from figure_utils import (
    COLOR_ALTERNATE,
    COLOR_CENTRAL,
    get_publication_colors,
    save_publication_figure,
    set_publication_style,
)


def main():
    set_publication_style()
    colors = get_publication_colors()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # 1. Mode A: central bar.
    ax.text(1.5, 4.5, "Central bar\nmode ($A$)", ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_CENTRAL,
                      alpha=0.2, edgecolor=COLOR_CENTRAL))

    y_schem = np.linspace(-1, 1, 50)
    zb_A = np.cos(np.pi * y_schem / 2) ** 2
    ax_mini_A = ax.inset_axes([1.0, 5.2, 1.0, 0.8], transform=ax.transData)
    ax_mini_A.plot(y_schem, zb_A, color=COLOR_CENTRAL, lw=2)
    ax_mini_A.text(0.5, 1.1, "bed", transform=ax_mini_A.transAxes,
                   ha="center", fontsize=7)
    ax_mini_A.axis("off")

    # 2. Mean flow distortion.
    ax.text(5.0, 3.0, "Mean flow\ndistortion ($\\delta u_0$)", ha="center",
            va="center", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="gray",
                      alpha=0.1, edgecolor="gray"))

    u0_dist = -0.5 * np.cos(np.pi * y_schem)
    ax_mini_U = ax.inset_axes([4.5, 3.7, 1.0, 0.8], transform=ax.transData)
    ax_mini_U.plot(y_schem, u0_dist, "k--", lw=1.5)
    ax_mini_U.axvline(0, color="gray", lw=0.5)
    ax_mini_U.text(0.5, 1.1, r"$\delta u_0$", transform=ax_mini_U.transAxes,
                   ha="center", fontsize=7)
    ax_mini_U.axis("off")

    # 3. Mode B: alternate bar.
    ax.text(8.5, 1.5, "Alternate bar\nmode ($B$)", ha="center", va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor=COLOR_ALTERNATE,
                      alpha=0.2, edgecolor=COLOR_ALTERNATE))

    zb_B = np.sin(np.pi * y_schem / 2)
    ax_mini_B = ax.inset_axes([8.0, 2.2, 1.0, 0.8], transform=ax.transData)
    ax_mini_B.plot(y_schem, zb_B, color=COLOR_ALTERNATE, lw=2)
    ax_mini_B.text(0.5, 1.1, "bed", transform=ax_mini_B.transAxes,
                   ha="center", fontsize=7)
    ax_mini_B.axis("off")

    # Edge A -> mean flow.
    ax.annotate("", xy=(4.2, 3.0), xytext=(2.3, 4.5),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5,
                                connectionstyle="arc3,rad=0.15"))
    ax.text(3.0, 3.9, "Generates", ha="center", fontsize=8, rotation=-25,
            backgroundcolor="white")

    # Edge mean flow -> B.
    ax.annotate("", xy=(7.7, 1.5), xytext=(5.8, 3.0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5,
                                connectionstyle="arc3,rad=0.15"))
    ax.text(6.8, 2.0, "Modifies stability\n(cross-enhancement)", ha="center",
            fontsize=8, rotation=-25)

    # Edge B -> A: competitive feedback closing the loop.
    ax.annotate("", xy=(1.5, 3.9), xytext=(8.0, 1.2),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2,
                                connectionstyle="arc3,rad=-0.3"))
    ax.text(4.5, 1.3, "Suppresses\n(competitive feedback)", color="black",
            ha="center", va="center", fontsize=8, rotation=-18,
            backgroundcolor="white")

    ax.text(8.5, 0.5, r"Growth rate $\uparrow$", ha="center",
            color=COLOR_ALTERNATE, fontsize=10, fontweight="bold")

    plt.tight_layout()
    save_publication_figure(fig, "Fig7_physical_mechanism")
    print("Figure 7 written.")


if __name__ == "__main__":
    main()
