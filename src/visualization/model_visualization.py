"""
Model visualization functions for domain and material properties
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path


def plot_material_curves(materials_dict, filename=None, figsize=(20, 4)):
    """
    Plot hydraulic characteristic curves for materials with parameters display
    
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
    fig, axes = plt.subplots(1, 4, figsize=figsize)
    
    # Pressure head range (from -10 to 0 m) - increased resolution for smoother curves
    hp_range = np.linspace(-10, 0, 1000)
    
    # Create text box for material parameters
    param_text = "SOIL PARAMETERS\n\n"
    
    for mat_name, material in materials_dict.items():
        hydraulic = material.hydraulic
        
        # Extract parameters based on material type
        param_text += f"{mat_name}:\n"
        param_text += f"  Ks = {material.Ks:.2e} m/s\n"
        param_text += f"  φ = {material.porosity:.3f} -\n"
        
        # Add hydraulic model specific parameters
        if hasattr(hydraulic, 'params'):  # VanGenuchtenModel
            params = hydraulic.params
            param_text += f"  Model: Van Genuchten\n"
            param_text += f"  θs = {params.theta_s:.3f} m³/m³\n"
            param_text += f"  θr = {params.theta_r:.3f} m³/m³\n" 
            param_text += f"  α = {params.alpha:.4f} m⁻¹\n"
            param_text += f"  n = {params.n:.3f} -\n"
            param_text += f"  m = {params.m:.3f} -\n"
        elif hasattr(hydraulic, '_theta_r'):  # CurveBasedHydraulicModel
            param_text += f"  Model: Curve-based\n"
            param_text += f"  θs = {hydraulic._theta_s:.3f} m³/m³\n"
            param_text += f"  θr = {hydraulic._theta_r:.3f} m³/m³\n"
            param_text += f"  ε = {hydraulic.epsilon:.4f} m\n"
            param_text += f"  Ss = {hydraulic.Ss:.2e} 1/m\n"
        else:
            param_text += f"  Model: Unknown type\n"
        
        param_text += "\n"
        
        # Plot curves based on model type
        if hasattr(hydraulic, 'params'):  # VanGenuchtenModel - analytical curves
            # Water content curve (smooth analytical)
            theta_values = [hydraulic._theta(hp) for hp in hp_range]
            axes[0].plot(hp_range, theta_values, label=mat_name, linewidth=2.5)
            
            # Relative permeability curve (smooth analytical)
            kr_values = [hydraulic._kr(hp) for hp in hp_range]
            axes[1].plot(hp_range, kr_values, label=mat_name, linewidth=2.5)
            
            # Effective saturation curve (smooth analytical)
            se_values = [hydraulic._Se(hp) for hp in hp_range]
            axes[2].plot(hp_range, se_values, label=mat_name, linewidth=2.5)
            
        elif hasattr(hydraulic, '_theta_interp'):  # CurveBasedHydraulicModel - from data points
            # Plot original data points + interpolated curves
            
            # Water content: show both data points and interpolation
            theta_values = [hydraulic._theta(hp) for hp in hp_range]
            axes[0].plot(hp_range, theta_values, label=f"{mat_name} (interp)", linewidth=2.5)
            
            # Show original data points if available
            if hasattr(hydraulic._theta_interp, 'curve'):
                curve = hydraulic._theta_interp.curve
                axes[0].scatter(curve.x_values, curve.y_values, 
                              marker='o', s=40, alpha=0.7, label=f"{mat_name} (data)")
            
            # Relative permeability: show both data points and interpolation
            kr_values = [hydraulic._kr(hp) for hp in hp_range]
            axes[1].plot(hp_range, kr_values, label=f"{mat_name} (interp)", linewidth=2.5)
            
            # Show original data points if available
            if hasattr(hydraulic._kr_interp, 'curve'):
                curve = hydraulic._kr_interp.curve
                axes[1].scatter(curve.x_values, curve.y_values, 
                              marker='o', s=40, alpha=0.7, label=f"{mat_name} (data)")
            
            # Effective saturation (computed from theta)
            se_values = [(hydraulic._theta(hp) - hydraulic._theta_r) / 
                        (hydraulic._theta_s - hydraulic._theta_r) for hp in hp_range]
            axes[2].plot(hp_range, se_values, label=mat_name, linewidth=2.5)
            
        else:
            # Fallback for unknown model types
            try:
                theta_values = [hydraulic._theta(hp) for hp in hp_range]
                kr_values = [hydraulic._kr(hp) for hp in hp_range]
                se_values = [hydraulic._Se(hp) for hp in hp_range]
                
                axes[0].plot(hp_range, theta_values, label=mat_name, linewidth=2.5)
                axes[1].plot(hp_range, kr_values, label=mat_name, linewidth=2.5)
                axes[2].plot(hp_range, se_values, label=mat_name, linewidth=2.5)
            except Exception as e:
                print(f"Warning: Could not plot curves for {mat_name}: {e}")
    
    # Water Content subplot
    axes[0].set_xlabel('Pressure Head Hp (m)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Water Content θ (m/m³)', fontsize=11, fontweight='bold')
    axes[0].set_title('Water Content Curve', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[0].set_xlim(-10, 0)
    
    # Relative Permeability subplot
    axes[1].set_xlabel('Pressure Head Hp (m)', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Relative Permeability Kr (-)', fontsize=11, fontweight='bold')
    axes[1].set_title('Relative Permeability Curve', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)
    axes[1].set_xlim(-10, 0)
    axes[1].set_ylim(0, 1.05)
    
    # Van Genuchten (Se) subplot
    axes[2].set_xlabel('Pressure Head Hp (m)', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Effective Saturation Se (-)', fontsize=11, fontweight='bold')
    axes[2].set_title('Effective Saturation Curve', fontsize=12, fontweight='bold')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(fontsize=9)
    axes[2].set_xlim(-10, 0)
    axes[2].set_ylim(0, 1.05)
    
    # Parameters text display
    axes[3].axis('off')
    axes[3].text(0.05, 0.95, param_text, transform=axes[3].transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    axes[3].set_title('Material Parameters', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"✓ Material curves saved to {filename}")
    
    return fig


def plot_domain_geometry(domain, boundary_conditions=None, water_table_level=None, filename=None, figsize=(14, 5)):
    """
    Plot domain geometry with material zones and water table
    
    Parameters:
    -----------
    domain : Domain
        Domain object with mesh and material assignments
    boundary_conditions : BoundaryConditionManager, optional
        Boundary conditions manager to get water table levels
    water_table_level : float, optional
        Fixed water table elevation in meters (overrides boundary_conditions)
    filename : str or Path, optional
        Output filename for saving the figure
    figsize : tuple
        Figure size (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get domain extents
    Lx, Ly = domain.Lx, domain.Ly
    
    # Define material colors - make them more visible
    material_colors = {
        'base': '#8B4513',      # SaddleBrown for till (more visible)
        'GI': '#228B22',        # ForestGreen for terreau/GI (more visible)
    }
    
    # Draw base material (entire domain)
    base_rect = patches.Rectangle(
        (0, 0), Lx, Ly,
        linewidth=2, edgecolor='black',
        facecolor=material_colors['base'],
        alpha=0.7, label='Till'
    )
    ax.add_patch(base_rect)
    
    # Draw material regions based on the new Domain structure
    for region_name in domain.regions:
        if region_name == 'base':
            continue  # Already drawn
            
        # For GI/rectangular regions, we need to reconstruct the bounds
        # by examining which coordinates are in the region
        mask = domain.regions[region_name]
        coords_in_region = domain.coords[mask]
        
        if len(coords_in_region) > 0:
            x_min, x_max = coords_in_region[:, 0].min(), coords_in_region[:, 0].max()
            y_min, y_max = coords_in_region[:, 1].min(), coords_in_region[:, 1].max()
            
            # Draw rectangle for this region
            width = x_max - x_min
            height = y_max - y_min
            
            color = material_colors.get(region_name, '#32CD32')  # LimeGreen default
            
            rect = patches.Rectangle(
                (x_min, y_min), width, height,
                linewidth=3, edgecolor='darkgreen',
                facecolor=color,
                alpha=0.8, label='Terreau'
            )
            ax.add_patch(rect)
            
            # Add label in center of rectangle
            cx = (x_min + x_max) / 2
            cy = (y_min + y_max) / 2
            ax.text(cx, cy, region_name.upper(), ha='center', va='center',
                   fontsize=14, fontweight='bold', color='white',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='darkgreen',
                            edgecolor='white', linewidth=2, alpha=0.9))
    
    # Plot water table from boundary conditions
    if boundary_conditions is not None:
        # Import here to avoid circular imports
        from .result_plotter import ResultsPlotter
        plotter = ResultsPlotter(domain=domain)
        plotter.plot_water_table(ax, t=0.0, bc_manager=boundary_conditions, domain=domain)
    
    # Draw constant water table if provided and no BC manager
    elif water_table_level is not None:
        ax.axhline(y=water_table_level, color='blue', linestyle='--',
                  linewidth=3, label=f'Water Table ({water_table_level:.1f}m)', alpha=0.8)

    
    ax.set_xlabel('x (m)', fontsize=13, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=13, fontweight='bold')
    ax.set_title('Domain Geometry - Project Setup',
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
