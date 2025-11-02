"""
Complete End-to-End Example: Parameter Optimization Workflow
Demonstrates the full pipeline from data loading to optimization
"""

from datetime import datetime, timedelta
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Import our optimization framework
from parameter_optimizer import (
    RichardsParameterOptimizer,
    ParameterBounds,
    ParameterVector
)
from data_preparation import (
    prepare_observation_data,
    quality_check_observations,
    remove_outliers,
    add_observation_weights
)
from optimize_soil_parameters import build_simulation_with_params


def compare_materials():
    """
    Compare different soil materials and their hydraulic properties
    Useful for setting realistic parameter bounds
    """
    print("="*70)
    print("TYPICAL VAN GENUCHTEN PARAMETERS FOR DIFFERENT SOIL TYPES")
    print("="*70)
    
    soils = {
        'Sand': {
            'theta_r': 0.045, 'theta_s': 0.430, 'alpha': 0.145, 'n': 2.68, 'Ks': 712.8
        },
        'Loamy Sand': {
            'theta_r': 0.057, 'theta_s': 0.410, 'alpha': 0.124, 'n': 2.28, 'Ks': 350.2
        },
        'Sandy Loam': {
            'theta_r': 0.065, 'theta_s': 0.410, 'alpha': 0.075, 'n': 1.89, 'Ks': 106.1
        },
        'Loam': {
            'theta_r': 0.078, 'theta_s': 0.430, 'alpha': 0.036, 'n': 1.56, 'Ks': 24.96
        },
        'Silt Loam': {
            'theta_r': 0.067, 'theta_s': 0.450, 'alpha': 0.020, 'n': 1.41, 'Ks': 10.8
        },
        'Clay Loam': {
            'theta_r': 0.095, 'theta_s': 0.410, 'alpha': 0.019, 'n': 1.31, 'Ks': 6.24
        },
        'Clay': {
            'theta_r': 0.068, 'theta_s': 0.380, 'alpha': 0.008, 'n': 1.09, 'Ks': 4.80
        }
    }
    
    print(f"\n{'Soil Type':<15} {'θ_r':<8} {'θ_s':<8} {'α':<8} {'n':<8} {'Ks (cm/day)':<12}")
    print("-"*70)
    
    for soil_name, params in soils.items():
        print(f"{soil_name:<15} "
              f"{params['theta_r']:<8.3f} "
              f"{params['theta_s']:<8.3f} "
              f"{params['alpha']:<8.3f} "
              f"{params['n']:<8.2f} "
              f"{params['Ks']:<12.1f}")
    
    print("\nReference: Carsel & Parrish (1988)")
    print("="*70)
    
    return soils


def create_bounds_from_soil_type(soil_type: str, variation: float = 0.3) -> ParameterBounds:
    """
    Create parameter bounds based on soil type with some variation
    
    Args:
        soil_type: Name of soil type (e.g., 'Sandy Loam')
        variation: Fractional variation around typical values (e.g., 0.3 = ±30%)
    
    Returns:
        ParameterBounds object
    """
    soils = compare_materials()
    
    if soil_type not in soils:
        raise ValueError(f"Unknown soil type: {soil_type}. Choose from: {list(soils.keys())}")
    
    typical = soils[soil_type]
    
    bounds = ParameterBounds(
        theta_r=(
            max(0.01, typical['theta_r'] * (1 - variation)),
            typical['theta_r'] * (1 + variation)
        ),
        theta_s=(
            typical['theta_s'] * (1 - variation),
            min(0.55, typical['theta_s'] * (1 + variation))
        ),
        alpha=(
            typical['alpha'] * (1 - variation),
            typical['alpha'] * (1 + variation)
        ),
        n=(
            max(1.1, typical['n'] * (1 - variation)),
            typical['n'] * (1 + variation)
        ),
        Ks=(
            typical['Ks'] * (1 - variation),
            typical['Ks'] * (1 + variation)
        )
    )
    
    print(f"\nCreated bounds for {soil_type}:")
    print(f"  θ_r:  [{bounds.theta_r[0]:.3f}, {bounds.theta_r[1]:.3f}]")
    print(f"  θ_s:  [{bounds.theta_s[0]:.3f}, {bounds.theta_s[1]:.3f}]")
    print(f"  α:    [{bounds.alpha[0]:.4f}, {bounds.alpha[1]:.4f}]")
    print(f"  n:    [{bounds.n[0]:.2f}, {bounds.n[1]:.2f}]")
    print(f"  Ks:   [{bounds.Ks[0]:.1f}, {bounds.Ks[1]:.1f}] cm/day")
    
    return bounds


def multi_start_optimization(
    observations,
    bounds,
    simulation_builder,
    n_starts: int = 5,
    material_zone: str = "base"
):
    """
    Run optimization from multiple starting points to avoid local minima
    
    Args:
        observations: ObservationData
        bounds: ParameterBounds
        simulation_builder: Function to build simulation
        n_starts: Number of random starting points
        material_zone: Material zone to optimize
    
    Returns:
        Best parameters and list of all results
    """
    print("\n" + "="*70)
    print(f"MULTI-START OPTIMIZATION ({n_starts} starting points)")
    print("="*70)
    
    param_names = ['theta_r', 'theta_s', 'alpha', 'n', 'Ks']
    all_results = []
    
    for i in range(n_starts):
        print(f"\n--- Starting Point {i+1}/{n_starts} ---")
        
        # Generate random initial parameters within bounds
        initial_params = {}
        for name in param_names:
            min_val, max_val = getattr(bounds, name)
            initial_params[name] = np.random.uniform(min_val, max_val)
        
        print(f"Initial guess: {initial_params}")
        
        # Create optimizer
        optimizer = RichardsParameterOptimizer(
            simulation_builder=simulation_builder,
            observations=observations,
            bounds=bounds,
            initial_params=initial_params,
            material_zone=material_zone
        )
        
        # Optimize
        try:
            best_params = optimizer.optimize(
                max_iterations=30,
                learning_rate=0.05,
                optimizer_type='adam',
                use_finite_diff=True
            )
            
            final_loss = optimizer.loss_history[-1]
            
            all_results.append({
                'initial': initial_params,
                'optimized': best_params,
                'loss': final_loss,
                'optimizer': optimizer
            })
            
            print(f"Final loss: {final_loss:.6f}")
            
        except Exception as e:
            print(f"Optimization failed: {e}")
            continue
    
    # Find best result
    best_result = min(all_results, key=lambda x: x['loss'])
    
    print("\n" + "="*70)
    print("MULTI-START RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n{'Run':<5} {'Final Loss':<15} {'θ_r':<8} {'θ_s':<8} {'α':<8} {'n':<8} {'Ks':<10}")
    print("-"*70)
    
    for i, result in enumerate(all_results):
        p = result['optimized']
        print(f"{i+1:<5} {result['loss']:<15.6f} "
              f"{p['theta_r']:<8.4f} {p['theta_s']:<8.4f} "
              f"{p['alpha']:<8.5f} {p['n']:<8.3f} {p['Ks']:<10.2f}")
    
    print(f"\nBest result from run {all_results.index(best_result) + 1}")
    print(f"Final loss: {best_result['loss']:.6f}")
    
    return best_result, all_results


def parameter_sensitivity_plot(
    observations,
    bounds,
    base_params: dict,
    simulation_builder,
    material_zone: str = "base"
):
    """
    Create sensitivity plots showing how each parameter affects the objective
    """
    print("\n" + "="*70)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*70)
    
    param_names = ['theta_r', 'theta_s', 'alpha', 'n', 'Ks']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, param_name in enumerate(param_names):
        print(f"\nAnalyzing {param_name}...")
        
        min_val, max_val = getattr(bounds, param_name)
        param_values = np.linspace(min_val, max_val, 15)
        losses = []
        
        for val in param_values:
            # Create test parameters
            test_params = base_params.copy()
            test_params[param_name] = val
            
            # Run simulation
            try:
                probe_manager = simulation_builder(test_params, material_zone)
                
                # Compute loss
                total_loss = 0.0
                for i, t in enumerate(observations.times):
                    for j, (x, y) in enumerate(observations.locations):
                        sim = probe_manager.get_value_at(t, x, y)
                        obs = observations.water_levels[i, j]
                        total_loss += (sim - obs) ** 2
                
                losses.append(total_loss)
            except:
                losses.append(np.nan)
        
        # Plot
        ax = axes[idx]
        ax.plot(param_values, losses, 'b-', linewidth=2)
        ax.axvline(base_params[param_name], color='r', linestyle='--', 
                  label='Optimized value')
        ax.set_xlabel(param_name)
        ax.set_ylabel('Objective Function')
        ax.set_title(f'Sensitivity to {param_name}')
        ax.legend()
        ax.grid(True)
    
    # Remove extra subplot
    fig.delaxes(axes[-1])
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nSensitivity analysis complete!")


def create_comprehensive_report(
    optimizer,
    best_params: dict,
    observations,
    soil_type: str,
    output_dir: Path = Path("optimization_results")
):
    """
    Create comprehensive optimization report with plots and statistics
    """
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*70)
    print("GENERATING COMPREHENSIVE REPORT")
    print("="*70)
    
    # 1. Basic optimization plots
    print("\n1. Creating optimization convergence plots...")
    optimizer.plot_results(save_path=output_dir / "optimization_convergence.png")
    
    # 2. Parameter comparison with typical values
    print("2. Comparing with typical soil properties...")
    soils = compare_materials()
    typical_params = soils[soil_type]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    param_names = list(best_params.keys())
    x = np.arange(len(param_names))
    width = 0.35
    
    # Normalize for visualization
    typical_norm = [typical_params[p] for p in param_names]
    optimized_norm = [best_params[p] for p in param_names]
    
    ax.bar(x - width/2, typical_norm, width, label=f'Typical {soil_type}', alpha=0.7)
    ax.bar(x + width/2, optimized_norm, width, label='Optimized', alpha=0.7)
    
    ax.set_xlabel('Parameter')
    ax.set_ylabel('Value')
    ax.set_title(f'Optimized vs Typical Parameters ({soil_type})')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "parameter_comparison.png", dpi=300)
    plt.close()
    
    # 3. Create text report
    print("3. Writing text report...")
    report_path = output_dir / "optimization_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("SOIL PARAMETER OPTIMIZATION REPORT\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Soil Type: {soil_type}\n\n")
        
        f.write("OPTIMIZED PARAMETERS:\n")
        f.write("-"*70 + "\n")
        for param, value in best_params.items():
            typical = typical_params[param]
            diff_pct = (value - typical) / typical * 100
            f.write(f"{param:10s}: {value:10.6f}  "
                   f"(typical: {typical:.6f}, diff: {diff_pct:+6.1f}%)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("OPTIMIZATION STATISTICS:\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of iterations: {len(optimizer.loss_history)}\n")
        f.write(f"Initial loss: {optimizer.loss_history[0]:.6f}\n")
        f.write(f"Final loss: {optimizer.loss_history[-1]:.6f}\n")
        f.write(f"Loss reduction: {(1 - optimizer.loss_history[-1]/optimizer.loss_history[0])*100:.1f}%\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("OBSERVATION DATA:\n")
        f.write("-"*70 + "\n")
        f.write(f"Number of observations: {len(observations.times)}\n")
        f.write(f"Number of locations: {len(observations.locations)}\n")
        f.write(f"Time range: {observations.times[0]/3600:.1f} to {observations.times[-1]/3600:.1f} hours\n")
        
        for i, loc in enumerate(observations.locations):
            wl = observations.water_levels[:, i]
            f.write(f"\nLocation {i} {loc}:\n")
            f.write(f"  Mean water level: {np.mean(wl):.3f} m\n")
            f.write(f"  Std deviation: {np.std(wl):.3f} m\n")
            f.write(f"  Range: [{np.min(wl):.3f}, {np.max(wl):.3f}] m\n")
    
    print(f"\nReport saved to: {output_dir}")
    print(f"  - optimization_convergence.png")
    print(f"  - parameter_comparison.png")
    print(f"  - optimization_report.txt")


def main():
    """
    Complete end-to-end workflow example
    """
    print("="*70)
    print("COMPLETE PARAMETER OPTIMIZATION WORKFLOW")
    print("="*70)
    
    # Step 1: Compare soil types and select
    print("\nStep 1: Reviewing soil types...")
    compare_materials()
    
    soil_type = "Sandy Loam"  # Choose based on your site
    print(f"\nSelected soil type: {soil_type}")
    
    # Step 2: Create bounds
    print("\nStep 2: Creating parameter bounds...")
    bounds = create_bounds_from_soil_type(soil_type, variation=0.4)
    
    # Step 3: Load and prepare observations
    print("\nStep 3: Loading observation data...")
    # For this example, we'll create synthetic data
    # In practice, use: obs = prepare_observation_data(csv_path=...)
    
    from optimize_soil_parameters import create_synthetic_observations
    
    soils = compare_materials()
    true_params = soils[soil_type]
    
    observation_times = np.linspace(0, 60*24*3600, 100)  # 60 days
    observation_locations = [(5.0, 1.0), (10.0, 1.0)]
    
    observations = create_synthetic_observations(
        true_params,
        observation_times,
        observation_locations,
        noise_level=0.02
    )
    
    # Step 4: Quality check
    print("\nStep 4: Quality checking observations...")
    qc_results = quality_check_observations(observations, plot=False)
    
    # Step 5: Multi-start optimization
    print("\nStep 5: Running multi-start optimization...")
    best_result, all_results = multi_start_optimization(
        observations,
        bounds,
        build_simulation_with_params,
        n_starts=3,
        material_zone="base"
    )
    
    # Step 6: Sensitivity analysis
    print("\nStep 6: Parameter sensitivity analysis...")
    parameter_sensitivity_plot(
        observations,
        bounds,
        best_result['optimized'],
        build_simulation_with_params,
        material_zone="base"
    )
    
    # Step 7: Generate comprehensive report
    print("\nStep 7: Generating comprehensive report...")
    create_comprehensive_report(
        best_result['optimizer'],
        best_result['optimized'],
        observations,
        soil_type
    )
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)
    print("\nFinal optimized parameters:")
    for param, value in best_result['optimized'].items():
        true_val = true_params[param]
        error = abs(value - true_val) / true_val * 100
        print(f"  {param:10s}: {value:8.5f}  (true: {true_val:8.5f}, error: {error:5.1f}%)")


if __name__ == "__main__":
    main()