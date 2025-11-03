"""
Utility functions for parameter optimization
Data loading, plotting, correlation analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from .optimizer import ObservationData, ParameterOptimizer


def load_observations_from_csv(
    csv_path: str,
    time_column: str = 'time',
    value_column: str = 'value',
    location: Tuple[float, float] = (0.0, 0.0),
    time_unit: str = 'seconds',
    start_time: Optional[float] = None
) -> ObservationData:
    """
    Load observation data from CSV
    
    Args:
        csv_path: Path to CSV file
        time_column: Name of time column
        value_column: Name of value column
        location: (x, y) coordinates of observation point
        time_unit: 'seconds', 'hours', 'days'
        start_time: Optional start time (will compute relative times)
    
    Returns:
        ObservationData
    """
    df = pd.read_csv(csv_path)
    
    # Parse times
    if time_unit == 'hours':
        times = df[time_column].values * 3600
    elif time_unit == 'days':
        times = df[time_column].values * 86400
    else:
        times = df[time_column].values
    
    # Make relative to start_time if provided
    if start_time is not None:
        times = times - start_time
    
    values = df[value_column].values.reshape(-1, 1)
    
    return ObservationData(
        times=times,
        locations=[location],
        values=values
    )


def plot_optimization_results(
    optimizer: ParameterOptimizer,
    true_params: Optional[Dict[str, float]] = None,
    save_path: Optional[str] = None
):
    """
    Plot optimization convergence and fit quality
    
    Args:
        optimizer: Fitted ParameterOptimizer
        true_params: Optional true parameters for comparison
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Loss convergence
    ax = axes[0, 0]
    ax.semilogy(optimizer.loss_history, 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Loss (log scale)', fontsize=12)
    ax.set_title('Optimization Convergence', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 2. Parameter evolution
    ax = axes[0, 1]
    for param_name in optimizer.param_names:
        values = [p[param_name] for p in optimizer.param_history]
        ax.plot(values, label=param_name, linewidth=2)
        
        # Add true value if provided
        if true_params and param_name in true_params:
            ax.axhline(true_params[param_name], linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Parameter Value', fontsize=12)
    ax.set_title('Parameter Evolution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. Observed vs Simulated
    best_params = optimizer.param_history[np.argmin(optimizer.loss_history)]
    best_sim = optimizer.forward_model(best_params)
    
    ax = axes[1, 0]
    times_hours = optimizer.observations.times / 3600
    ax.plot(times_hours, optimizer.observations.values[:, 0], 
            'ko', label='Observed', markersize=6)
    ax.plot(times_hours, best_sim[:, 0], 
            'r-', label='Simulated', linewidth=2)
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Value', fontsize=12)
    ax.set_title('Best Fit', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Residuals
    residuals = best_sim[:, 0] - optimizer.observations.values[:, 0]
    ax = axes[1, 1]
    ax.plot(times_hours, residuals, 'bo-', linewidth=2)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Residual', fontsize=12)
    ax.set_title('Residuals', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    
    print("\nFit Statistics:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  Loss: {optimizer.loss_history[-1]:.6e}")


def compute_parameter_correlation(
    optimizer: ParameterOptimizer,
    n_samples: int = 100,
    plot: bool = True
) -> np.ndarray:
    """
    Compute parameter correlation matrix by sampling parameter space
    Helps identify if parameters are uniquely identifiable
    
    Args:
        optimizer: ParameterOptimizer instance
        n_samples: Number of samples
        plot: Whether to plot correlation matrix
    
    Returns:
        Correlation matrix
    """
    print(f"Sampling {n_samples} parameter combinations...")
    
    param_samples = []
    losses = []
    
    for i in range(n_samples):
        # Random parameters within bounds
        params = {}
        for name in optimizer.param_names:
            min_val, max_val = optimizer.bounds.get(name)
            params[name] = np.random.uniform(min_val, max_val)
        
        # Evaluate
        try:
            simulated = optimizer.forward_model(params)
            loss = optimizer.compute_loss(simulated)
            
            param_samples.append([params[name] for name in optimizer.param_names])
            losses.append(loss)
        except:
            continue
    
    param_array = np.array(param_samples)
    correlation = np.corrcoef(param_array.T)
    
    # Find high correlations
    print("\nParameter Correlations:")
    for i, name1 in enumerate(optimizer.param_names):
        for j in range(i+1, len(optimizer.param_names)):
            name2 = optimizer.param_names[j]
            corr = correlation[i, j]
            if abs(corr) > 0.7:
                print(f"  {name1} ↔ {name2}: r = {corr:+.3f} ⚠️")
    
    if plot:
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(correlation, cmap='RdBu_r', vmin=-1, vmax=1)
        
        ax.set_xticks(range(len(optimizer.param_names)))
        ax.set_yticks(range(len(optimizer.param_names)))
        ax.set_xticklabels(optimizer.param_names, rotation=45, ha='right')
        ax.set_yticklabels(optimizer.param_names)
        ax.set_title('Parameter Correlation Matrix', fontsize=14, fontweight='bold')
        
        # Add values
        for i in range(len(optimizer.param_names)):
            for j in range(len(optimizer.param_names)):
                ax.text(j, i, f'{correlation[i, j]:.2f}',
                       ha="center", va="center", fontsize=10)
        
        plt.colorbar(im, ax=ax, label='Correlation')
        plt.tight_layout()
        plt.savefig('parameter_correlation.png', dpi=300)
        plt.show()
    
    return correlation


def estimate_computational_cost(
    n_params: int,
    n_iterations: int,
    simulation_time_sec: float,
    method: str = 'both'
) -> Dict[str, float]:
    """
    Estimate computational cost for optimization
    
    Args:
        n_params: Number of parameters to optimize
        n_iterations: Number of iterations
        simulation_time_sec: Time for one forward simulation
        method: 'finite_diff', 'adjoint', or 'both'
    
    Returns:
        Dictionary with time estimates
    """
    results = {}
    
    if method in ['finite_diff', 'both']:
        fd_sims = (n_params + 1) * n_iterations
        fd_time = fd_sims * simulation_time_sec
        results['finite_diff'] = {
            'simulations': fd_sims,
            'time_seconds': fd_time,
            'time_hours': fd_time / 3600
        }
    
    if method in ['adjoint', 'both']:
        adj_sims = 2.5 * n_iterations
        adj_time = adj_sims * simulation_time_sec
        results['adjoint'] = {
            'simulations': adj_sims,
            'time_seconds': adj_time,
            'time_hours': adj_time / 3600
        }
    
    if method == 'both':
        results['speedup'] = results['finite_diff']['time_hours'] / results['adjoint']['time_hours']
    
    # Print summary
    print(f"\nComputational Cost Estimate:")
    print(f"  Parameters: {n_params}")
    print(f"  Iterations: {n_iterations}")
    print(f"  Simulation time: {simulation_time_sec:.1f} sec")
    print()
    
    if 'finite_diff' in results:
        fd = results['finite_diff']
        print(f"Finite Differences:")
        print(f"  Simulations: {fd['simulations']:.0f}")
        print(f"  Time: {fd['time_hours']:.1f} hours")
    
    if 'adjoint' in results:
        adj = results['adjoint']
        print(f"\nAdjoint Method:")
        print(f"  Simulations: {adj['simulations']:.0f}")
        print(f"  Time: {adj['time_hours']:.1f} hours")
    
    if 'speedup' in results:
        print(f"\nSpeedup: {results['speedup']:.1f}x")
    
    return results


def print_parameter_comparison(
    optimized: Dict[str, float],
    initial: Optional[Dict[str, float]] = None,
    true: Optional[Dict[str, float]] = None
):
    """
    Print formatted comparison of parameter values
    
    Args:
        optimized: Optimized parameters
        initial: Initial parameters (optional)
        true: True parameters (optional)
    """
    print("\n" + "="*70)
    print("PARAMETER RESULTS")
    print("="*70)
    
    # Header
    header = f"{'Parameter':<15}"
    if true:
        header += f"{'True':<12}"
    if initial:
        header += f"{'Initial':<12}"
    header += f"{'Optimized':<12}"
    if true:
        header += f"{'Error %':<10}"
    
    print(header)
    print("-"*70)
    
    # Values
    for name in optimized.keys():
        row = f"{name:<15}"
        
        if true:
            row += f"{true[name]:<12.6f}"
        if initial:
            row += f"{initial[name]:<12.6f}"
        row += f"{optimized[name]:<12.6f}"
        
        if true:
            error = abs(optimized[name] - true[name]) / true[name] * 100
            row += f"{error:<10.2f}"
        
        print(row)
    
    print("="*70)