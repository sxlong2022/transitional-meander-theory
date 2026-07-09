"""
Figure 7: Physical Mechanism Schematic
Conceptual diagram of the nonlinear cross-enhancement mechanism.
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

from figure_utils import set_publication_style, save_for_publication, get_publication_colors

def main():
    set_publication_style()
    colors = get_publication_colors()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Define styles
    box_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1.5)
    arrow_props = dict(arrowstyle='->', color='black', lw=1.5, connectionstyle='arc3,rad=0.0')
    
    # ================= NODES =================
    
    # 1. Mode A (Central bar) - Left position
    ax.text(1.5, 4.5, "Central bar\nmode ($A$)", ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[0], alpha=0.2, edgecolor=colors[0]))
    
    # Decorate with a small schematic of Central Bar (Above node)
    y_schem = np.linspace(-1, 1, 50)
    zb_A = np.cos(np.pi * y_schem)**2  # Schematic symmetric hump
    # Using bounds [x, y, width, height] in Data coordinates
    ax_mini_A = ax.inset_axes([1.0, 5.2, 1.0, 0.8], transform=ax.transData)
    ax_mini_A.plot(y_schem, zb_A, color=colors[0], lw=2)
    ax_mini_A.text(0.5, 1.1, "bed", transform=ax_mini_A.transAxes, ha='center', fontsize=7)
    ax_mini_A.axis('off')
    
    # (Self-Interaction label removed - detail explained in caption)

    # 3. Mean flow distortion - Center
    ax.text(5.0, 3.0, "Mean flow\ndistortion ($\delta u_0$)", ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='gray', alpha=0.1, edgecolor='gray'))
            
    # Decorate with schematic of flow distortion (Above node)
    u0_dist = -0.5 * np.cos(np.pi * y_schem) # Schematic distortion
    ax_mini_U = ax.inset_axes([4.5, 3.7, 1.0, 0.8], transform=ax.transData)
    ax_mini_U.plot(y_schem, u0_dist, 'k--', lw=1.5)
    ax_mini_U.axvline(0, color='gray', lw=0.5)
    ax_mini_U.text(0.5, 1.1, r"$\delta u_0$", transform=ax_mini_U.transAxes, ha='center', fontsize=7)
    ax_mini_U.axis('off')

    # 4. Mode B (Alternate bar) - Bottom Right
    ax.text(8.5, 1.5, "Alternate bar\nmode ($B$)", ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[1], alpha=0.2, edgecolor=colors[1]))

    # Decorate with schematic of Alternate Bar (Above node)
    zb_B = np.sin(np.pi * y_schem) # Schematic antisymmetric
    ax_mini_B = ax.inset_axes([8.0, 2.2, 1.0, 0.8], transform=ax.transData)
    ax_mini_B.plot(y_schem, zb_B, color=colors[1], lw=2)
    ax_mini_B.text(0.5, 1.1, "bed", transform=ax_mini_B.transAxes, ha='center', fontsize=7)
    ax_mini_B.axis('off')
    
    # ================= EDGES / ARROWS =================
    
    # Path 1: A -> Mean Flow (from right edge of A to left edge of Mean Flow)
    # Mode A center: (1.5, 4.5), Mean Flow center: (5.0, 3.0)
    ax.annotate("", xy=(4.2, 3.0), xytext=(2.3, 4.5),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle='arc3,rad=0.15'))
    ax.text(3.0, 3.9, "Generates", ha='center', fontsize=8, rotation=-25, backgroundcolor='white')
    
    # Path 2: Mean Flow -> B (from right edge of Mean Flow to left edge of B)
    # Mean Flow center: (5.0, 3.0), Mode B center: (8.5, 1.5)
    ax.annotate("", xy=(7.7, 1.5), xytext=(5.8, 3.0),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5, connectionstyle='arc3,rad=0.15'))
    ax.text(6.8, 2.0, "Modifies stability\n(cross-enhancement)", ha='center', fontsize=8, rotation=-25)

    # Path 3: Direct Feedback (optional, maybe distracting, let's keep it creating a loop feeling)
    # Let's show B benefiting
    ax.text(8.5, 0.5, r"Growth rate $\uparrow$", ha='center', color=colors[6], fontsize=10, fontweight='bold')
    
    # (Summary text removed to avoid occlusion - detail is in caption)

    plt.tight_layout()
    save_for_publication(fig, 'Fig7_physical_mechanism')
    print("Figure 7 saved.")

if __name__ == "__main__":
    main()
