"""Shared figure styling and export helpers for the reproducibility package.

All rendered figures are written to ``figures/`` at the repository root.  The
directory is created on demand, so a fresh clone needs no manual setup.  Vector
PDF is the primary manuscript asset; a 300 dpi PNG preview is written alongside
it for quick inspection.
"""

import os

import matplotlib.pyplot as plt

# Directory layout: <repo>/code/figure_utils.py -> <repo>/figures/
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(CODE_DIR)
FIGURES_DIR = os.path.join(REPO_ROOT, "figures")

# Colorblind-safe palette (Okabe-Ito).
COLOR_CYCLE = [
    "#E69F00",  # Orange
    "#56B4E9",  # Sky Blue
    "#009E73",  # Bluish Green
    "#F0E442",  # Yellow
    "#0072B2",  # Blue
    "#D55E00",  # Vermilion
    "#CC79A7",  # Reddish Purple
    "#000000",  # Black
]

# Canonical semantic colors, held identical across every figure.
COLOR_CENTRAL = "#0072B2"    # mode A, central bar (m = 2)
COLOR_ALTERNATE = "#CC79A7"  # mode B, alternate bar (m = 1)
COLOR_MIXED = "#009E73"      # mixed state / co-evolving


def get_publication_colors():
    """Return the Okabe-Ito color cycle as a list of hex strings."""
    return COLOR_CYCLE


def get_semantic_colors():
    """Return the canonical semantic colors keyed by bar mode."""
    return {
        "central": COLOR_CENTRAL,
        "alternate": COLOR_ALTERNATE,
        "mixed": COLOR_MIXED,
    }


def save_publication_figure(fig, filename, verbose=True):
    """Write a figure to ``figures/`` as PDF and PNG.

    Args:
        fig: Matplotlib figure object.
        filename: Base name without extension, for example ``'Fig4_phase_plane'``.
        verbose: Print the written paths when ``True``.

    Returns:
        ``(pdf_path, png_path)`` of the written files.
    """
    base_name = os.path.splitext(filename)[0]
    os.makedirs(FIGURES_DIR, exist_ok=True)

    pdf_path = os.path.join(FIGURES_DIR, f"{base_name}.pdf")
    png_path = os.path.join(FIGURES_DIR, f"{base_name}.png")

    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    if verbose:
        print(f"Saved PDF: {pdf_path}")
        print(f"Saved PNG: {png_path}")

    return pdf_path, png_path


def set_publication_style():
    """Apply the publication rcParams used for every manuscript figure."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 10,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.titlesize": 12,
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.5,
    })
