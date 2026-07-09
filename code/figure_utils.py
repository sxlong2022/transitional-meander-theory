import os
import shutil
import matplotlib.pyplot as plt

# Configuration

# Colorblind Safe Color Palette (Colorblind Safe - Okabe-Ito)
# https://jfly.uni-koeln.de/color/
COLOR_CYCLE = [
    '#E69F00', # Orange
    '#56B4E9', # Sky Blue
    '#009E73', # Bluish Green
    '#F0E442', # Yellow
    '#0072B2', # Blue
    '#D55E00', # Vermilion
    '#CC79A7', # Reddish Purple
    '#000000'  # Black
]

def get_publication_colors():
    """Return list of colorblind-safe hex codes."""
    return COLOR_CYCLE

def save_for_publication(fig, filename, verbose=True):
    """
    Save figure locally as vector PDF and raster PNG preview.
    """
    base_name = os.path.splitext(filename)[0]
    pdf_name = f"{base_name}.pdf"
    png_name = f"{base_name}.png"
    
    fig.savefig(pdf_name, dpi=600, bbox_inches='tight')
    fig.savefig(png_name, dpi=300, bbox_inches='tight')
    
    if verbose:
        print(f"[Local] Saved figure to: {os.path.abspath(pdf_name)}")
        print(f"[Local] Saved preview to: {os.path.abspath(png_name)}")

def set_publication_style():
    """
    Set matplotlib rcParams for Publication style (Times New Roman, legible font sizes).
    """
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'text.usetex': False, # Set to True if you have LaTeX installed
        'mathtext.fontset': 'stix',
        'lines.linewidth': 1.0,
        'axes.linewidth': 0.5
    })
