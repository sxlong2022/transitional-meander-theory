"""Figure 1: problem configuration for a variable-width transitional channel.

Panel (a) shows the plan view: a straight channel whose half-width ``b(s)`` first
diverges (``db/ds > 0``, ``sigma > 0``) and then converges (``db/ds < 0``,
``sigma < 0``).  Central bars occupy the diverging reach and alternate bars form
in the transition toward the converging reach.  Panel (b) shows representative
cross-sectional bed profiles for the symmetric central bar mode (``m = 2``), the
antisymmetric alternate bar mode (``m = 1``), and their superposition in the mixed
state.  Each cross-section spans the normalized transverse coordinate
``zeta in [-1, 1]``.

Run from the repository root::

    python code/fig01_conceptual_sketch.py
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

_CODE_DIR = os.path.dirname(os.path.abspath(__file__))
if _CODE_DIR not in sys.path:
    sys.path.insert(0, _CODE_DIR)

from figure_utils import (
    COLOR_ALTERNATE,
    COLOR_CENTRAL,
    COLOR_MIXED,
    get_publication_colors,
    save_publication_figure,
    set_publication_style,
)


def draw_channel_planview(ax, colors):
    """Draw the plan view of a variable-width straight channel with bars."""
    x = np.linspace(0, 12, 200)
    y_center = np.zeros_like(x)

    # Half-width grows to a maximum near x = 6, then contracts.
    W0 = 0.5
    dW = 0.3
    half_width = W0 + dW * np.sin(np.pi * x / 12)

    y_top = y_center + half_width
    y_bot = y_center - half_width

    ax.fill_between(x, y_bot, y_top, color="lightblue", alpha=0.3)
    ax.plot(x, y_top, "k-", lw=1.5)
    ax.plot(x, y_bot, "k-", lw=1.5)

    # Central bar: symmetric mid-channel deposit in the diverging reach.
    cb = Ellipse((5.0, 0), width=2.4, height=0.4, color=COLOR_CENTRAL, alpha=0.7)
    ax.add_patch(cb)
    ax.text(5.0, -0.65, "Central bar", fontsize=8, ha="center",
            color=COLOR_CENTRAL, fontweight="bold")

    # Alternate bars: diagonal, bank-attached, alternating sides.
    ab1_x = [8.0, 9.5, 10.0, 8.5]
    ab1_y = [0.55, 0.45, 0.25, 0.35]
    ax.fill(ab1_x, ab1_y, color=COLOR_ALTERNATE, alpha=0.7)

    ab2_x = [9.5, 11.0, 11.5, 10.0]
    ab2_y = [-0.45, -0.35, -0.15, -0.25]
    ax.fill(ab2_x, ab2_y, color=COLOR_ALTERNATE, alpha=0.7)
    ax.text(9.8, 0.75, "Alternate bars", fontsize=8, ha="center",
            color=COLOR_ALTERNATE, fontweight="bold")

    ax.annotate("", xy=(11.8, 0), xytext=(0.2, 0),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
    ax.text(0.5, -0.25, "Flow", fontsize=9, color="black", ha="left")

    # Half-width annotation at the widest section.
    x_w = 6
    w_val = W0 + dW * np.sin(np.pi * x_w / 12)
    ax.annotate("", xy=(x_w + 0.3, w_val), xytext=(x_w + 0.3, 0),
                arrowprops=dict(arrowstyle="<->", color="gray", lw=1))
    ax.text(x_w + 0.5, w_val / 2, r"$b(s)$", fontsize=9, va="center", color="gray")

    # Reach labels keyed to the sign of db/ds.
    ax.text(3.0, 1.15, r"Diverging reach ($\mathrm{d}b/\mathrm{d}s > 0$)",
            fontsize=8.5, style="italic", ha="center", fontweight="bold")
    ax.text(9.0, 1.15, r"Converging reach ($\mathrm{d}b/\mathrm{d}s < 0$)",
            fontsize=8.5, style="italic", ha="center", fontweight="bold")

    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.2, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(a) Plan view: variable-width channel", fontsize=10, loc="left")


def draw_cross_sections(ax, colors):
    """Draw the cross-sectional bed profiles of the three morphologies."""
    y = np.linspace(-1, 1, 100)

    # Central bar: even profile, crest on the channel axis.
    zb_central = 0.4 * np.exp(-4 * y**2)
    ax.plot(y - 2.5, zb_central, color=COLOR_CENTRAL, lw=2, label="Central bar")
    ax.fill_between(y - 2.5, 0, zb_central, color=COLOR_CENTRAL, alpha=0.3)
    ax.text(-2.5, 0.55, "Central bar", fontsize=8, ha="center",
            color=COLOR_CENTRAL, fontweight="bold")

    # Alternate bar: odd profile, nodal line on the channel axis.
    zb_alternate = 0.4 * np.sin(np.pi * y)
    ax.plot(y, zb_alternate, color=COLOR_ALTERNATE, lw=2, label="Alternate bar")
    ax.fill_between(y, 0, zb_alternate, where=(zb_alternate > 0),
                    color=COLOR_ALTERNATE, alpha=0.3)
    ax.fill_between(y, 0, zb_alternate, where=(zb_alternate < 0),
                    color=COLOR_ALTERNATE, alpha=0.3)
    ax.text(0, 0.55, "Alternate bar", fontsize=8, ha="center",
            color=COLOR_ALTERNATE, fontweight="bold")

    # Mixed state: superposition of the two modes.
    zb_mixed = 0.3 * np.exp(-4 * y**2) + 0.35 * np.sin(np.pi * y)
    ax.plot(y + 2.5, zb_mixed, color=COLOR_MIXED, lw=2, label="Mixed state")
    ax.fill_between(y + 2.5, 0, zb_mixed, where=(zb_mixed > 0),
                    color=COLOR_MIXED, alpha=0.3)
    ax.fill_between(y + 2.5, 0, zb_mixed, where=(zb_mixed < 0),
                    color=COLOR_MIXED, alpha=0.3)
    ax.text(2.5, 0.55, "Mixed state", fontsize=8, ha="center",
            color=COLOR_MIXED, fontweight="bold")

    ax.axhline(0, color="k", lw=0.5, ls="--")

    for offset in [-2.5, 0, 2.5]:
        ax.plot([offset - 1, offset - 1], [-0.1, 0.1], "k-", lw=2)
        ax.plot([offset + 1, offset + 1], [-0.1, 0.1], "k-", lw=2)

    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.6, 0.7)
    ax.set_ylabel("Bed elevation", fontsize=9)
    ax.set_xlabel(r"Normalized transverse coordinate $\zeta$", fontsize=9)
    ax.set_title("(b) Cross-sectional bed profiles", fontsize=10, loc="left")
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(["-1", "0", "1"])


def main():
    set_publication_style()
    colors = get_publication_colors()

    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), height_ratios=[1.2, 1])

    draw_channel_planview(axes[0], colors)
    draw_cross_sections(axes[1], colors)

    plt.tight_layout()
    save_publication_figure(fig, "Fig1_conceptual_sketch")
    print("Figure 1 (conceptual sketch) written.")


if __name__ == "__main__":
    main()
