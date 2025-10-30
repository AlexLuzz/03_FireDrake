"""
Model visualization functions for domain and material properties
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def plot_material_curves(materials_dict, filename=None, figsize=(15, 4)):
    """
    Plot hydraulic characteristic curves for materials
    
    Parameters:
    -----------
    materials_dict : dict
        Dictionary with material names as keys and Material objects as values
        e.g., {"Till": till_mat, "Terreau": terreau_mat}
    filename : str or Path, optional
        Output filename for saving the figure
    figsize : tuple
        Figure size (width, height)
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Pressure head range (from -10 to 0 m)
    hp_range = np.linspace(-10, 0, 200)
    
    for mat_name, material in materials_dict.items():
        # Water content curve
        theta_values = [material.hydraulic._theta(hp) for hp in hp_range]
        axes[0].plot(hp_range, theta_values, label=mat_name, linewidth=2)
        
        # Relative permeability curve
        kr_values = [material.hydraulic._kr(hp) for hp in hp_range]
        axes[1].plot(hp_range, kr_values, label=mat_name, linewidth=2)
        
        # Effective saturation curve
        se_values = [material.hydraulic._Se(hp) for hp in hp_range]
        axes[2].plot(hp_range, se_values, label=mat_name, linewidth=2)
    
    # Water Content subplot
    axes[0].set_xlabel('Pressure Head Hp (m)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Water Content θ (m/m³)', fontsize=11, fontweight='bold')
    axes[0].set_title('Water Content Curve', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=10)
    axes[0].set_xlim(-10, 0)
    
    # Relative Permeability subplot
    axes[1].set_xlabel('Pressure Head Hp (m)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Relative Permeability Kr (-)', fontsize=11, fontweight='bold')
    axes[1].set_title('Relative Permeability Curve', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=10)
    axes[1].set_xlim(-10, 0)
    axes[1].set_ylim(0, 1.05)
    
    # Van Genuchten (Se) subplot
    axes[2].set_xlabel('Pressure Head Hp (m)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Effective Saturation Se (-)', fontsize=11, fontweight='bold')
    axes[2].set_title('Van Genuchten Curve (Se)', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=10)
    axes[2].set_xlim(-10, 0)
    axes[2].set_ylim(0, 1.05)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Material curves saved to {filename}")
    
    return fig


def plot_domain_geometry(domain, water_table_level=None, filename=None, figsize=(14, 5)):
    """
    Plot domain geometry with material zones
    
    Parameters:
    -----------
    domain : Domain
        Domain object with mesh and material assignments
    water_table_level : float, optional
        Water table elevation in meters (for visualization)
    filename : str or Path, optional
        Output filename for saving the figure
    figsize : tuple
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get domain extents
    Lx, Ly = domain.Lx, domain.Ly
    
    # Define material colors
    material_colors = {
        'base': '#CD853F',      # Peru/tan for till
        'GI': '#90EE90',        # Light green for terreau/GI
    }
    
    # Draw base material (entire domain)
    base_rect = patches.Rectangle(
        (0, 0), Lx, Ly,
        linewidth=2, edgecolor='black',
        facecolor=material_colors['base'],
        hatch='///', label='Till'
    )
    ax.add_patch(base_rect)
    
    # Draw any sub-rectangles (like GI zone)
    if hasattr(domain, 'rectangles') and domain.rectangles:
        for rect_name, rect_data in domain.rectangles.items():
            x_min, x_max = rect_data['x_min'], rect_data['x_max']
            y_min, y_max = rect_data['y_min'], rect_data['y_max']
            width = x_max - x_min
            height = y_max - y_min
            
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=2, edgecolor='darkgreen',
                facecolor=material_colors.get(rect_name, '#90EE90'),
                label='Terreau'
            )
            ax.add_patch(rect)
            
            # Add label in center of rectangle
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            ax.text(cx, cy, rect_name.upper(), ha='center', va='center',
                   fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                            edgecolor='darkgreen', linewidth=2))
    
    # Draw water table if provided
    if water_table_level is not None:
        ax.axhline(y=water_table_level, color='blue', linestyle='--',
                  linewidth=2, label=f'Water Table ({water_table_level}m)')
        
        # Add saturated/unsaturated zone labels
        mid_x = Lx / 2
        if water_table_level > Ly / 2:
            # Water table in upper half
            ax.text(mid_x, Ly * 0.7, 'UNSATURATED\nZONE',
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.7', facecolor='wheat',
                            edgecolor='black', linewidth=1.5, alpha=0.8))
            ax.text(mid_x, Ly * 0.2, 'SATURATED\nZONE',
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue',
                            edgecolor='black', linewidth=1.5, alpha=0.8))
        else:
            # Water table in lower half
            ax.text(mid_x, Ly * 0.8, 'UNSATURATED\nZONE',
                   ha='center', va='center', fontsize=16, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.7', facecolor='wheat',
                            edgecolor='black', linewidth=1.5, alpha=0.8))
            if water_table_level > 0.5:  # Only show if visible
                ax.text(mid_x, water_table_level / 2, 'SATURATED\nZONE',
                       ha='center', va='center', fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.7', facecolor='lightblue',
                                edgecolor='black', linewidth=1.5, alpha=0.8))
    
    ax.set_xlabel('x (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=13, fontweight='bold')
    ax.set_title('Material Distribution - Project Domain',
                fontsize=14, fontweight='bold')
    ax.set_xlim(0, Lx)
    ax.set_ylim(0, Ly)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Domain geometry saved to {filename}")
    
    return fig
