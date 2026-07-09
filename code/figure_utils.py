import os
import shutil
import matplotlib.pyplot as plt

# Configuration
LATEX_TEMPLATE_DIR = r"D:\Temp\JFM_LaTeX_Template_2"

# JFM/Cambridge University Press Color Palette (Colorblind Safe - Okabe-Ito)
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

def get_jfm_colors():
    """Return list of colorblind-safe hex codes."""
    return COLOR_CYCLE

def save_for_jfm(fig, filename, verbose=True):
    """
    Save figure locally (PDF & PNG) and copy PDF to JFM LaTeX template directory.

    Args:
        fig: matplotlib figure object
        filename: string (e.g., 'Fig2') - extension will be added automatically
        verbose: bool, whether to print status
    """
    # Ensure no extension in base name
    base_name = os.path.splitext(filename)[0]

    # 1. Save Locally (PDF for vector, PNG for preview)
    pdf_name = f"{base_name}.pdf"
    png_name = f"{base_name}.png"

    fig.savefig(pdf_name, dpi=600, bbox_inches='tight')
    fig.savefig(png_name, dpi=300, bbox_inches='tight')

    if verbose:
        print(f"[Local] Saved figure to: {os.path.abspath(pdf_name)}")

    # 2. Copy to LaTeX Directory (PDF only)
    if os.path.exists(LATEX_TEMPLATE_DIR):
        target_path = os.path.join(LATEX_TEMPLATE_DIR, pdf_name)
        try:
            shutil.copy2(pdf_name, target_path)
            if verbose:
                print(f"[Publication] Copied to: {target_path}")
        except Exception as e:
            print(f"[Error] Failed to copy to {LATEX_TEMPLATE_DIR}: {e}")
    else:
        # Only warn if verbose to avoid clutter if user knows dirt doesn't exist
        if verbose:
            print(f"[Warning] LaTeX template directory not found: {LATEX_TEMPLATE_DIR}")

def set_jfm_style():
    """
    Set matplotlib rcParams for JFM style (Times New Roman, legible font sizes).
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
