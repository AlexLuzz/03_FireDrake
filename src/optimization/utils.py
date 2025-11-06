"""
Utility functions for adjoint-based parameter optimization
Plotting, analysis, and result visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
from .optimizer import AdjointOptimizer, create_parameter_controls, ObservationData
from pyadjoint import pause_annotation, continue_annotation


def extract_probe_data(probe_manager, field_name: str = 'water_table'):
    """
    Extract probe data from ProbeManager and convert to numpy array
    
    Args:
        probe_manager: ProbeManager object with recorded data
        field_name: Name of field to extract (default: 'water_table')
    
    Returns:
        numpy array of shape [n_times, n_probes]
    """
    probe_names = probe_manager.names
    n_probes = len(probe_names)
    
    # Get data from first probe to determine n_times
    first_probe_data = probe_manager.get_probe_data(probe_names[0], field_name)
    n_times = len(first_probe_data)
    
    # Build array [n_times, n_probes]
    result = np.zeros((n_times, n_probes))
    for i, probe_name in enumerate(probe_names):
        result[:, i] = probe_manager.get_probe_data(probe_name, field_name)
    
    return result


def plot_optimization_results(
    optimizer,
    forward_model_func,
    observations: ObservationData,
    initial_params: Dict[str, float],
    optimized_params: Dict[str, float],
    mesh,
    domain,
    V,
    save_path: Optional[str] = None
):
    """
    Plot optimization convergence and fit quality for adjoint optimization
    
    Args:
        optimizer: Fitted AdjointOptimizer with loss_history and param_history
        forward_model_func: Your simulation function that takes (param_constants, domain, V)
        observations: ObservationData object
        initial_params: Initial parameter values dict
        optimized_params: Optimized parameter values dict
        mesh: Firedrake mesh for creating parameter controls
        domain: Domain object for simulation
        V: FunctionSpace for simulation
        save_path: Optional path to save figure
    """
    # Run simulations to get data for plotting
    print("Generating plots (running initial and optimized simulations)...")
    
    # Initial simulation
    _, initial_constants = create_parameter_controls(initial_params, mesh)
    pause_annotation()  # Don't record these on tape
    initial_sim_pm = forward_model_func(initial_constants, domain, V)
    
    # Optimized simulation  
    _, optimized_constants = create_parameter_controls(optimized_params, mesh)
    optimized_sim_pm = forward_model_func(optimized_constants, domain, V)
    continue_annotation()
    
    # Extract data from ProbeManagers
    initial_sim = extract_probe_data(initial_sim_pm)
    optimized_sim = extract_probe_data(optimized_sim_pm)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # -------------------------------------------------------------------------
    # 1. Loss convergence
    # -------------------------------------------------------------------------
    ax = axes[0, 0]
    iterations = range(len(optimizer.loss_history))
    ax.semilogy(iterations, optimizer.loss_history, 'b-', linewidth=2, marker='o')
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add improvement percentage
    improvement = (1 - optimizer.loss_history[-1]/optimizer.loss_history[0]) * 100
    ax.text(0.05, 0.95, f'Improvement: {improvement:.1f}%',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # -------------------------------------------------------------------------
    # 2. Parameter evolution
    # -------------------------------------------------------------------------
    ax = axes[0, 1]
    n_params = len(optimizer.control_names)
    colors = plt.cm.tab10(np.linspace(0, 1, n_params))
    
    for i, param_name in enumerate(optimizer.control_names):
        values = [p[param_name] for p in optimizer.param_history]
        # Normalize to show relative changes
        initial_val = values[0]
        normalized = [(v - initial_val) / initial_val * 100 for v in values]
        ax.plot(iterations, normalized, label=param_name, 
                linewidth=2, color=colors[i])
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Change from Initial (%)', fontsize=12)
    ax.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=8, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color='k', linestyle='--', alpha=0.3)
    
    # -------------------------------------------------------------------------
    # 3. Gradient norms (if available)
    # -------------------------------------------------------------------------
    ax = axes[0, 2]
    # For adjoint, we don't have explicit gradient history, 
    # but we can show loss reduction rate
    if len(optimizer.loss_history) > 1:
        loss_reductions = [-np.diff(optimizer.loss_history)]
        ax.plot(range(1, len(optimizer.loss_history)), 
                loss_reductions[0], 'g-', linewidth=2, marker='o')
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Loss Reduction', fontsize=12)
        ax.set_title('Optimization Progress', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='r', linestyle='--', alpha=0.5)
    else:
        ax.text(0.5, 0.5, 'Need >1 iteration\nfor progress plot',
                ha='center', va='center', transform=ax.transAxes)
    
    # -------------------------------------------------------------------------
    # 4-6. Time series for each probe
    # -------------------------------------------------------------------------
    times_hours = observations.times * 24  # Convert days to hours
    n_probes = observations.values.shape[1]
    probe_names = ['Probe 1 (x=8.0m)', 'Probe 2 (x=10.0m)', 'Probe 3 (x=12.5m)']
    
    for probe_idx in range(min(3, n_probes)):
        ax = axes[1, probe_idx]
        
        # Observed data
        ax.plot(times_hours, observations.values[:, probe_idx],
                'ko', label='Observed', markersize=4, alpha=0.6)
        
        # Initial simulation
        ax.plot(times_hours, initial_sim[:, probe_idx],
                'b--', label='Initial', linewidth=2, alpha=0.7)
        
        # Optimized simulation
        ax.plot(times_hours, optimized_sim[:, probe_idx],
                'r-', label='Optimized', linewidth=2)
        
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Water Table (m)', fontsize=12)
        ax.set_title(probe_names[probe_idx], fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Calculate RMSE for optimized
        residuals = optimized_sim[:, probe_idx] - observations.values[:, probe_idx]
        rmse = np.sqrt(np.mean(residuals**2))
        ax.text(0.95, 0.05, f'RMSE: {rmse:.3f}m',
                transform=ax.transAxes, fontsize=10,
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    plt.show()
    
    # -------------------------------------------------------------------------
    # Print summary statistics
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("FIT STATISTICS")
    print("="*70)
    
    for probe_idx in range(n_probes):
        probe_name = f"Probe {probe_idx + 1}"
        
        # Initial fit
        initial_residuals = initial_sim[:, probe_idx] - observations.values[:, probe_idx]
        initial_rmse = np.sqrt(np.mean(initial_residuals**2))
        initial_mae = np.mean(np.abs(initial_residuals))
        
        # Optimized fit
        optimized_residuals = optimized_sim[:, probe_idx] - observations.values[:, probe_idx]
        optimized_rmse = np.sqrt(np.mean(optimized_residuals**2))
        optimized_mae = np.mean(np.abs(optimized_residuals))
        
        print(f"\n{probe_name}:")
        print(f"  Initial   - RMSE: {initial_rmse:.4f}, MAE: {initial_mae:.4f}")
        print(f"  Optimized - RMSE: {optimized_rmse:.4f}, MAE: {optimized_mae:.4f}")
        print(f"  Improvement: {(1 - optimized_rmse/initial_rmse)*100:.1f}%")
    
    print("\n" + "="*70)


def print_parameter_comparison(
    optimized: Dict[str, float],
    initial: Dict[str, float],
    bounds: Optional[Dict[str, tuple]] = None
):
    """
    Print formatted comparison of parameter values
    
    Args:
        optimized: Optimized parameters
        initial: Initial parameters
        bounds: Optional bounds dict {name: (min, max)}
    """
    print("\n" + "="*70)
    print("PARAMETER COMPARISON")
    print("="*70)
    
    # Header
    if bounds:
        print(f"{'Parameter':<20} {'Initial':>12} {'Optimized':>12} {'Change %':>10} {'Status':>10}")
    else:
        print(f"{'Parameter':<20} {'Initial':>12} {'Optimized':>12} {'Change %':>10}")
    print("-"*70)
    
    # Values
    for name in optimized.keys():
        init_val = initial[name]
        opt_val = optimized[name]
        change = ((opt_val - init_val) / init_val) * 100
        
        row = f"{name:<20} {init_val:>12.6e} {opt_val:>12.6e} {change:>10.2f}"
        
        # Check if at bounds
        if bounds and name in bounds:
            min_b, max_b = bounds[name]
            if abs(opt_val - min_b) / min_b < 0.01:
                row += f"{'MIN BOUND':>10}"
            elif abs(opt_val - max_b) / max_b < 0.01:
                row += f"{'MAX BOUND':>10}"
            else:
                row += f"{'OK':>10}"
        
        print(row)
    
    print("="*70)


def plot_residual_analysis(
    optimizer: AdjointOptimizer,
    forward_model_func,
    observations,
    optimized_params: Dict[str, float],
    mesh,
    domain,
    V,
    save_path: Optional[str] = None
):
    """
    Detailed residual analysis for each probe
    
    Args:
        optimizer: AdjointOptimizer instance
        forward_model_func: Your simulation function that takes (param_constants, domain, V)
        observations: ObservationData object
        optimized_params: Optimized parameter values
        mesh: Firedrake mesh for creating parameter controls
        domain: Domain object for simulation
        V: FunctionSpace for simulation
        save_path: Optional save path
    """
    # Run optimized simulation
    _, opt_constants = create_parameter_controls(optimized_params, mesh)
    pause_annotation()
    optimized_sim_pm = forward_model_func(opt_constants, domain, V)
    continue_annotation()
    
    # Extract data from ProbeManager
    optimized_sim = extract_probe_data(optimized_sim_pm)
    
    times_hours = observations.times * 24  # Convert days to hours
    n_probes = observations.values.shape[1]
    
    fig, axes = plt.subplots(n_probes, 2, figsize=(14, 4*n_probes))
    if n_probes == 1:
        axes = axes.reshape(1, -1)
    
    for probe_idx in range(n_probes):
        residuals = optimized_sim[:, probe_idx] - observations.values[:, probe_idx]
        
        # Time series of residuals
        ax = axes[probe_idx, 0]
        ax.plot(times_hours, residuals, 'b-', linewidth=2)
        ax.axhline(0, color='k', linestyle='--')
        ax.fill_between(times_hours, residuals, 0, alpha=0.3)
        ax.set_xlabel('Time (hours)', fontsize=12)
        ax.set_ylabel('Residual (m)', fontsize=12)
        ax.set_title(f'Probe {probe_idx + 1} - Residuals Over Time', 
                     fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_res = np.max(np.abs(residuals))
        ax.text(0.02, 0.98, 
                f'RMSE: {rmse:.3f}m\nMAE: {mae:.3f}m\nMax: {max_res:.3f}m',
                transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        # Histogram of residuals
        ax = axes[probe_idx, 1]
        ax.hist(residuals, bins=30, density=True, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(0, color='r', linestyle='--', linewidth=2, label='Zero')
        ax.axvline(np.mean(residuals), color='g', linestyle='--', linewidth=2, 
                   label=f'Mean: {np.mean(residuals):.3f}')
        ax.set_xlabel('Residual (m)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Probe {probe_idx + 1} - Residual Distribution', 
                     fontsize=13, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Residual analysis saved to: {save_path}")
    plt.show()


def validate_optimized_parameters(
    forward_model_func,
    observations,
    optimized_params: Dict[str, float],
    mesh,
    domain,
    V,
    n_runs: int = 2
):
    """
    Validate optimized parameters by running multiple simulations
    Useful to check simulation stability/convergence
    
    Args:
        forward_model_func: Your simulation function that takes (param_constants, domain, V)
        observations: ObservationData object
        optimized_params: Optimized parameters to validate
        mesh: Firedrake mesh for creating parameter controls
        domain: Domain object for simulation
        V: FunctionSpace for simulation
        n_runs: Number of validation runs
    """
    print("\n" + "="*70)
    print("VALIDATION: Running multiple simulations with optimized parameters")
    print("="*70)
    
    _, opt_constants = create_parameter_controls(optimized_params, mesh)
    
    losses = []
    for i in range(n_runs):
        print(f"\nRun {i+1}/{n_runs}...")
        pause_annotation()
        sim_pm = forward_model_func(opt_constants, domain, V)
        continue_annotation()
        
        # Extract data from ProbeManager
        sim = extract_probe_data(sim_pm)
        
        # Compute loss
        cutoff = int(0.2 * len(sim))
        residuals = (sim[cutoff:] - observations.values[cutoff:]) * observations.weights[cutoff:]
        loss = np.sum(residuals ** 2) / np.sum(observations.weights[cutoff:])
        losses.append(loss)
        print(f"  Loss: {loss:.6e}")
    
    print(f"\nValidation Summary:")
    print(f"  Mean loss:   {np.mean(losses):.6e}")
    print(f"  Std loss:    {np.std(losses):.6e}")
    print(f"  Min loss:    {np.min(losses):.6e}")
    print(f"  Max loss:    {np.max(losses):.6e}")
    print(f"  Variability: {np.std(losses)/np.mean(losses)*100:.2f}%")
    
    if np.std(losses) / np.mean(losses) < 0.01:
        print("\n✓ Simulations are stable (low variability)")
    else:
        print("\n⚠ Simulations show variability - check convergence settings")
    
    print("="*70)