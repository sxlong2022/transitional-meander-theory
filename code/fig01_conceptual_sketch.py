"""
Figure 1: Conceptual Sketch - Transitional Meandering River
Shows the problem setup with variable-width channel and mixed bar morphology.
REVISED: Simplified straight channel (no sinuosity) for cleaner conceptual illustration.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse, Polygon
import sys
import os

from figure_utils import set_publication_style, get_publication_colors

def draw_channel_planview(ax, colors):
    """Draw a plan view of a variable-width straight channel with bars."""
    
    # Straight channel centerline at y=0
    x = np.linspace(0, 12, 200)
    y_center = np.zeros_like(x)  # Straight channel
    
    # Variable width: wide in middle (diverging), narrow at ends
    # Width profile: W(x) = W0 + dW * sin(pi*x/L)
    W0 = 0.5  # Base half-width
    dW = 0.3  # Maximum width variation
    half_width = W0 + dW * np.sin(np.pi * x / 12)
    
    y_top = y_center + half_width
    y_bot = y_center - half_width
    
    # Draw channel banks
    ax.fill_between(x, y_bot, y_top, color='lightblue', alpha=0.3)
    ax.plot(x, y_top, 'k-', lw=1.5)
    ax.plot(x, y_bot, 'k-', lw=1.5)
    
    # Draw Central Bar (symmetric, mid-channel deposit) in widest section (x≈6)
    cb = Ellipse((6, 0), width=2.5, height=0.4, color=colors[0], alpha=0.7)
    ax.add_patch(cb)
    ax.text(6, -0.65, 'Central bar', fontsize=8, ha='center', color=colors[0])
    
    # Draw Alternate Bars (diagonal, bank-attached) in transition zone (x=8-11)
    # Bar 1: attached to upper bank
    ab1_x = [8.0, 9.5, 10.0, 8.5]
    ab1_y = [0.55, 0.45, 0.25, 0.35]
    ax.fill(ab1_x, ab1_y, color=colors[1], alpha=0.7)
    
    # Bar 2: attached to lower bank (diagonal opposite)
    ab2_x = [9.5, 11.0, 11.5, 10.0]
    ab2_y = [-0.45, -0.35, -0.15, -0.25]
    ax.fill(ab2_x, ab2_y, color=colors[1], alpha=0.7)
    ax.text(10.5, 0.7, 'Alternate bars', fontsize=8, ha='center', color=colors[1])
    
    # Flow direction arrow (simple, along centerline)
    ax.annotate('', xy=(11.8, 0), xytext=(0.2, 0),
                arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    ax.text(0.5, -0.25, 'Flow', fontsize=9, color='blue', ha='left')
    
    # Width annotation at widest point (x=6)
    x_w = 6
    w_val = W0 + dW * np.sin(np.pi * x_w / 12)
    ax.annotate('', xy=(x_w + 0.3, w_val), xytext=(x_w + 0.3, -w_val),
                arrowprops=dict(arrowstyle='<->', color='gray', lw=1))
    ax.text(x_w + 0.5, 0.2, r'$B(s)$', fontsize=9, va='center', color='gray')
    
    # Zone labels at top
    ax.text(2, 1.1, 'Converging', fontsize=8, style='italic', ha='center')
    ax.text(6, 1.1, 'Diverging', fontsize=8, style='italic', ha='center')
    ax.text(10, 1.1, 'Transition zone', fontsize=8, style='italic', ha='center', fontweight='bold')
    
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1.2, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) Plan view: variable-width channel', fontsize=10, loc='left')


def draw_cross_sections(ax, colors):
    """Draw cross-sectional views of different bar types."""
    
    y = np.linspace(-1, 1, 100)
    
    # Cross-section 1: Central bar (symmetric)
    zb_central = 0.4 * np.exp(-4 * y**2)
    ax.plot(y - 2.5, zb_central, color=colors[0], lw=2, label='Central bar')
    ax.fill_between(y - 2.5, 0, zb_central, color=colors[0], alpha=0.3)
    ax.text(-2.5, 0.55, 'Central bar', fontsize=8, ha='center')
    
    # Cross-section 2: Alternate bar (antisymmetric)
    zb_alternate = 0.4 * np.sin(np.pi * y)
    ax.plot(y, zb_alternate, color=colors[1], lw=2, label='Alternate bar')
    ax.fill_between(y, 0, zb_alternate, where=(zb_alternate > 0), color=colors[1], alpha=0.3)
    ax.fill_between(y, 0, zb_alternate, where=(zb_alternate < 0), color=colors[1], alpha=0.3)
    ax.text(0, 0.55, 'Alternate bar', fontsize=8, ha='center')
    
    # Cross-section 3: Mixed state
    zb_mixed = 0.3 * np.exp(-4 * y**2) + 0.35 * np.sin(np.pi * y)
    ax.plot(y + 2.5, zb_mixed, color=colors[6], lw=2, label='Mixed state')
    ax.fill_between(y + 2.5, 0, zb_mixed, where=(zb_mixed > 0), color=colors[6], alpha=0.3)
    ax.fill_between(y + 2.5, 0, zb_mixed, where=(zb_mixed < 0), color=colors[6], alpha=0.3)
    ax.text(2.5, 0.55, 'Mixed state', fontsize=8, ha='center')
    
    # Baseline
    ax.axhline(0, color='k', lw=0.5, ls='--')
    
    # Bank markers
    for offset in [-2.5, 0, 2.5]:
        ax.plot([offset - 1, offset - 1], [-0.1, 0.1], 'k-', lw=2)
        ax.plot([offset + 1, offset + 1], [-0.1, 0.1], 'k-', lw=2)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-0.6, 0.7)
    ax.set_ylabel('Bed elevation', fontsize=9)
    ax.set_xlabel('Normalized transverse coordinate $\zeta$', fontsize=9)
    ax.set_title('(b) Cross-sectional bed profiles', fontsize=10, loc='left')
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['-1', '0', '1'])


def main():
    set_publication_style()
    colors = get_publication_colors()
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 4.5), height_ratios=[1.2, 1])
    
    draw_channel_planview(axes[0], colors)
    draw_cross_sections(axes[1], colors)
    
    plt.tight_layout()
    
    # Save as PNG and PDF (no SVG)
    local_dir = os.path.dirname(os.path.abspath(__file__))
    png_path = os.path.join(local_dir, 'Fig1_conceptual_sketch.png')
    pdf_path = os.path.join(local_dir, 'Fig1_conceptual_sketch.pdf')
    
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')
    
    print(f"[Local] Saved PNG to: {png_path}")
    print(f"[Local] Saved PDF to: {pdf_path}")
    
    # Print local save status
    print("\nFigure 1 (Conceptual Sketch) saved.")

if __name__ == "__main__":
    main()
