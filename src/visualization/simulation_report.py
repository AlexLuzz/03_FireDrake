"""
Report generator framework for simulation results
Define tools to create multi-page PDF reports with configuration, model setup, and results
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
from datetime import datetime
import numpy as np
from .basic_plotting import BasicPlotting

class SimulationReport(BasicPlotting):
    
    def __init__(self, plotter, output_dir=None):
        self.plotter = plotter
        self.config = self.plotter.config
        self.domain = self.plotter.domain
        self.bc_manager = self.plotter.bc_manager
        self.output_dir = self.config.output_dir if output_dir is None else output_dir
    
    def _add_config_text(self, config, domain, ax=None):
        """Add configuration text box to page"""
        ax.axis('off')
        
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


    def _add_contaminant_text(self, fig, gridspec_slice, contaminant_props):
        """Add contaminant properties text box"""
        ax = fig.add_subplot(gridspec_slice)
        ax.axis('off')
        
        contam_text = "CONTAMINANT PROPERTIES\n\n"
        for key, value in contaminant_props.items():
            contam_text += f"{key}: {value}\n"
        
        ax.text(0.1, 0.9, contam_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _add_material_curves(self, materials, axes=None):
        # Create curves in temporary figure
        self.plot_material_curves(materials, axes=axes)
        plt.close("all")
    
    def _add_domain_geometry(self, domain, boundary_conditions=None, ax=None):
        """
        Add domain geometry plot to page
        Creates temporary figure with geometry, then copies to main figure
        """
        # Extract region data from domain
        regions = {}
        for region_name in domain.regions:
            if region_name == 'base':
                continue
            mask = domain.regions[region_name]
            coords_in_region = domain.coords[mask]
            if len(coords_in_region) > 0:
                regions[region_name] = {
                    'x_bounds': (coords_in_region[:, 0].min(), coords_in_region[:, 0].max()),
                    'y_bounds': (coords_in_region[:, 1].min(), coords_in_region[:, 1].max()),
                    'label': 'Terreau' if region_name == 'GI' else region_name
                }
        
        self.plot_domain_geometry(domain.Lx, domain.Ly, regions, ax=ax)
        
        # Add water table if available
        if boundary_conditions:
            self.plot_water_table(ax, t=0.0, bc_manager=boundary_conditions, domain=domain)
        plt.close("all")

    def _create_timeseries_page(self, plotter, field_type, compare=True):
        """Create time series page using ResultsPlotter"""
        plotting_config = {
            'time_series_fields': [field_type],
            'plot_snapshots': False,
            'plot_comsol_comparison': compare,
            'plot_measured_comparison': compare
        }
        
        fig = plotter.plot_complete_results(plotting_config=plotting_config, return_figure=True)
        
        # Set appropriate title
        title_map = {
            'water_table': 'WATER TABLE ELEVATION - TIME SERIES',
            'concentration': 'CONTAMINANT CONCENTRATION - TIME SERIES',
            'pressure': 'PRESSURE HEAD - TIME SERIES'
        }
        fig.suptitle(title_map.get(field_type, f'{field_type.upper()} - TIME SERIES'),
                    fontsize=16, fontweight='bold', y=0.98)
        return fig
    
    def _create_snapshots_page(self, plotter, field_type):
        """Create spatial snapshots page using ResultsPlotter"""
        if not plotter.snapshot_manager:
            return self._create_no_snapshots_placeholder()
        
        plotting_config = {
            'time_series_fields': [],
            'plot_snapshots': True,
            'snapshot_fields': [field_type],
            'plot_comsol_comparison': False,
            'plot_measured_comparison': False
        }
        
        fig = plotter.plot_complete_results(plotting_config=plotting_config, return_figure=True)
        
        # Set appropriate title
        title_map = {
            'saturation': 'SOIL SATURATION - SPATIAL DISTRIBUTION',
            'concentration': 'CONTAMINANT CONCENTRATION - SPATIAL DISTRIBUTION',
            'pressure': 'PRESSURE HEAD - SPATIAL DISTRIBUTION'
        }
        fig.suptitle(title_map.get(field_type, f'{field_type.upper()} - SPATIAL DISTRIBUTION'),
                    fontsize=16, fontweight='bold', y=0.98)
        return fig

    def save_pdf(self, figures, filename, metadata=None):
        """
        Save dictionary of figures to PDF
        
        Parameters
        ----------
        figures : dict
            Dictionary with page names as keys and matplotlib figures as values
            Example: {'page1': fig1, 'page2': fig2}
        filename : str or Path
            Output PDF filename
        metadata : dict, optional
            PDF metadata (Title, Author, Subject, Keywords)
        """
        filepath = self.output_dir / filename
        
        # Default metadata
        if metadata is None:
            metadata = {
                'Title': 'Simulation Report',
                'Author': 'FireDrake Simulation',
                'Subject': 'Simulation Results',
                'Creator': 'FireDrake Framework',
                'CreationDate': datetime.now()
            }
        
        with PdfPages(filepath, metadata=metadata) as pdf:
            for page_name, fig in figures.items():
                pdf.savefig(fig, bbox_inches='tight')
        
        print(f"✓ Report saved: {filepath}")
        return filepath

