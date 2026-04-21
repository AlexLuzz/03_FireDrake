
def add_config_text(ax, config, domain):
    """Add configuration text box to page"""
    
    config_text = f"""
        SIMULATION CONFIGURATION

        Simulation Name: {config.project_name}
        Start: {config.start_datetime.strftime('%Y-%m-%d %H:%M')}
        End: {config.end_datetime.strftime('%Y-%m-%d %H:%M')}
        Duration: {config.t_end_hours:.1f} hours ({config.t_end/(24*3600):.1f} days)
        Time Step: {config.dt:.1f} seconds

        Domain: {domain.Lx} m × {domain.Ly} m | Mesh: {domain.nx} × {domain.ny}

        Simulation duration (real time): {config.real_time_duration}
        """
    ax.text(0.1, 0.95, config_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
