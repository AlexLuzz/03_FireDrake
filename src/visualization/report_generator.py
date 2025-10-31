"""
Report generator for simulation results
Creates multi-page PDF reports with configuration, model setup, and results
"""
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime
import numpy as np

from .model_visualization import plot_material_curves, plot_domain_geometry


class SimulationReport:
    """
    Generate comprehensive simulation reports
    Creates PDF reports with configuration parameters, model setup, and results
    """
    
    def __init__(self, output_dir=None):
        """
        Parameters:
        -----------
        output_dir : str or Path, optional
            Directory for saving reports (default: results/)
        """
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def print_richards_report(self, config, domain, field_map, plotter,
                            boundary_conditions=None, filename=None, include_snapshots=True,
                            plotting_config: dict = None):
        """
        Generate Richards equation simulation report (3 pages)
        
        Page 1: Configuration + Soil Parameters + Domain Geometry
        Page 2: Time series results with rainfall
        Page 3: Spatial snapshots (if available)
        
        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration object
        domain : Domain
            Domain with geometry and material assignments
        field_map : MaterialField
            Material field mapper
        plotter : ResultsPlotter
            Results plotter with probe and snapshot data
        boundary_conditions : BoundaryConditionManager, optional
            Boundary conditions for water table display
        filename : str or Path, optional
            Output PDF filename
        include_snapshots : bool
            Whether to include snapshot page
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"richards_report_{timestamp}.pdf"
        
        print(f"\n📄 Generating Richards report: {filename}")
        
        with PdfPages(filename) as pdf:
            # PAGE 1: Configuration + Model Setup
            self._create_richards_setup_page(config, domain, field_map, boundary_conditions)
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # PAGE 2: Time Series Results
            self._create_timeseries_page(plotter, field_type='water_table', plotting_config=plotting_config)
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # PAGE 3: Spatial Snapshots (if requested and available)
            if include_snapshots and plotter.snapshot_manager:
                self._create_snapshots_page(plotter, field_type='saturation', plotting_config=plotting_config)
                pdf.savefig(bbox_inches='tight')
                plt.close()
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Richards Equation Simulation Report'
            d['Author'] = 'FireDrake Simulation'
            d['Subject'] = f'Simulation: {config.name}'
            d['CreationDate'] = datetime.now()
        
        print(f"✓ Report saved: {filename}")
        return filename
    
    def print_transport_report(self, config, domain, field_map, plotter,
                              contaminant_props=None, filename=None,
                              include_richards=False):
        """
        Generate coupled transport simulation report (4-5 pages)
        
        Page 1: Configuration + Soil + Contaminant Parameters + Domain
        Page 2: Richards time series (optional)
        Page 3: Transport time series
        Page 4: Richards snapshots (optional)
        Page 5: Transport concentration snapshots
        
        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration object
        domain : Domain
            Domain with geometry and material assignments
        field_map : MaterialField
            Material field mapper with transport properties
        plotter : ResultsPlotter
            Results plotter with probe and snapshot data
        contaminant_props : dict, optional
            Contaminant properties to display
        filename : str or Path, optional
            Output PDF filename
        include_richards : bool
            Whether to include Richards equation results
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.output_dir / f"transport_report_{timestamp}.pdf"
        
        print(f"\n📄 Generating Transport report: {filename}")
        
        with PdfPages(filename) as pdf:
            # PAGE 1: Configuration + Model Setup (with transport params)
            self._create_transport_setup_page(config, domain, field_map, contaminant_props)
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # PAGE 2 (optional): Richards Time Series
            if include_richards:
                self._create_timeseries_page(plotter, field_type='water_table')
                pdf.savefig(bbox_inches='tight')
                plt.close()
            
            # PAGE 3: Transport Time Series
            self._create_timeseries_page(plotter, field_type='concentration')
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # PAGE 4 (optional): Richards Snapshots
            if include_richards and plotter.snapshot_manager:
                self._create_snapshots_page(plotter, field_type='saturation')
                pdf.savefig(bbox_inches='tight')
                plt.close()
            
            # PAGE 5: Transport Concentration Snapshots
            if plotter.snapshot_manager:
                self._create_snapshots_page(plotter, field_type='concentration')
                pdf.savefig(bbox_inches='tight')
                plt.close()
            
            # Add metadata
            d = pdf.infodict()
            d['Title'] = 'Coupled Flow-Transport Simulation Report'
            d['Author'] = 'FireDrake Simulation'
            d['Subject'] = f'Simulation: {config.name}'
            d['CreationDate'] = datetime.now()
        
        print(f"✓ Report saved: {filename}")
        return filename
    
    def _create_richards_setup_page(self, config, domain, field_map, boundary_conditions=None):
        """Create page 1 for Richards report: Config + Soil + Geometry"""
        fig = plt.figure(figsize=(11, 14))
        gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1.2, 0.1, 1], 
                              width_ratios=[1, 1, 1, 1], hspace=0.35, wspace=0.25)
        
        # Section 1: Configuration Parameters (spans all columns)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        config_text = f"""
SIMULATION CONFIGURATION

Simulation Name: {config.name}
Start Date: {config.start_datetime.strftime('%Y-%m-%d %H:%M')}
End Date: {config.end_datetime.strftime('%Y-%m-%d %H:%M')}
Duration: {config.t_end_hours:.1f} hours ({config.t_end/(24*3600):.1f} days)
Time Step: {config.dt:.1f} seconds ({config.dt/3600:.2f} hours)
Number of Steps: {config.num_steps}

Domain Size: {domain.Lx} m × {domain.Ly} m
Mesh Resolution: {domain.nx} × {domain.ny} elements
"""
        ax1.text(0.1, 0.95, config_text, transform=ax1.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Section 2: Material Curves (4 columns)
        # Get materials from domain
        materials = {}
        if hasattr(domain, 'materials'):
            materials = domain.materials
        
        # Plot material curves directly in the gridspec
        if materials:
            # Create the curves plot and extract data
            temp_fig = plot_material_curves(materials)
            
            # Create 4 subplots in row 1 (material curves + parameters)
            curve_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
            
            # Copy the curves to the main figure (now includes 4th subplot for parameters)
            for i, (ax_src, ax_dst) in enumerate(zip(temp_fig.axes, curve_axes)):
                if i < 3:  # First 3 are curves
                    # Copy plot content
                    for line in ax_src.get_lines():
                        ax_dst.plot(line.get_xdata(), line.get_ydata(),
                                  label=line.get_label(), linewidth=2)
                    ax_dst.set_xlabel(ax_src.get_xlabel(), fontweight='bold', fontsize=9)
                    ax_dst.set_ylabel(ax_src.get_ylabel(), fontweight='bold', fontsize=9)
                    ax_dst.set_title(ax_src.get_title(), fontweight='bold', fontsize=10)
                    ax_dst.grid(True, alpha=0.3)
                    ax_dst.legend(fontsize=8)
                    ax_dst.tick_params(labelsize=8)
                    if hasattr(ax_src, 'get_xlim'):
                        ax_dst.set_xlim(ax_src.get_xlim())
                    if hasattr(ax_src, 'get_ylim'):
                        ax_dst.set_ylim(ax_src.get_ylim())
                else:  # 4th subplot is parameters text
                    ax_dst.axis('off')
                    ax_dst.set_title(ax_src.get_title(), fontweight='bold', fontsize=10)
                    # Copy text elements
                    for text in ax_src.texts:
                        bbox_props = None
                        if text.get_bbox_patch():
                            bbox = text.get_bbox_patch()
                            bbox_props = dict(boxstyle=bbox.get_boxstyle(), 
                                             facecolor=bbox.get_facecolor(),
                                             edgecolor=bbox.get_edgecolor(),
                                             linewidth=bbox.get_linewidth(),
                                             alpha=bbox.get_alpha())
                        
                        ax_dst.text(text.get_position()[0], text.get_position()[1],
                                text.get_text(), ha=text.get_ha(), va=text.get_va(),
                                fontsize=text.get_fontsize()*0.8, fontfamily=text.get_fontfamily(),
                                bbox=bbox_props, transform=ax_dst.transAxes)
            
            plt.close(temp_fig)
        
        # Section 3: Domain Geometry (spans all columns, bottom section)
        # Plot domain geometry in a temporary figure and extract info
        temp_domain_fig = plot_domain_geometry(domain, boundary_conditions=boundary_conditions)
        
        # Create domain subplot and recreate content
        ax3 = fig.add_subplot(gs[3, :])
        temp_ax = temp_domain_fig.axes[0]
        
        # Recreate patches (rectangles) - can't copy patches between figures
        import matplotlib.patches as mpatches
        for patch in temp_ax.patches:
            # Create new patch with same properties
            if isinstance(patch, mpatches.Rectangle):
                new_patch = mpatches.Rectangle(
                    patch.get_xy(), patch.get_width(), patch.get_height(),
                    linewidth=patch.get_linewidth(),
                    edgecolor=patch.get_edgecolor(),
                    facecolor=patch.get_facecolor(),
                    hatch=patch.get_hatch(),
                    label=patch.get_label() if patch.get_label() != '_nolegend_' else None
                )
                ax3.add_patch(new_patch)
        
        # Copy lines (water table, etc.)
        for line in temp_ax.get_lines():
            ax3.plot(line.get_xdata(), line.get_ydata(),
                    color=line.get_color(), linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(), 
                    label=line.get_label() if line.get_label() != '_nolegend_' else None)
        
        # Copy text annotations
        for text in temp_ax.texts:
            bbox_props = None
            if text.get_bbox_patch():
                bbox = text.get_bbox_patch()
                bbox_props = dict(boxstyle=bbox.get_boxstyle(), 
                                 facecolor=bbox.get_facecolor(),
                                 edgecolor=bbox.get_edgecolor(),
                                 linewidth=bbox.get_linewidth(),
                                 alpha=bbox.get_alpha())
            
            ax3.text(text.get_position()[0], text.get_position()[1],
                    text.get_text(), ha=text.get_ha(), va=text.get_va(),
                    fontsize=text.get_fontsize(), fontweight=text.get_fontweight(),
                    bbox=bbox_props)
        
        # Copy axis properties
        ax3.set_xlabel(temp_ax.get_xlabel(), fontweight='bold', fontsize=11)
        ax3.set_ylabel(temp_ax.get_ylabel(), fontweight='bold', fontsize=11)
        ax3.set_title(temp_ax.get_title(), fontweight='bold', fontsize=12)
        ax3.set_xlim(temp_ax.get_xlim())
        ax3.set_ylim(temp_ax.get_ylim())
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.3)
        
        # Recreate legend
        if temp_ax.get_legend():
            handles, labels = ax3.get_legend_handles_labels()
            if handles:
                ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        plt.close(temp_domain_fig)
        
        fig.suptitle('RICHARDS EQUATION SIMULATION - MODEL SETUP',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def _create_transport_setup_page(self, config, domain, field_map, contaminant_props):
        """Create page 1 for Transport report: Config + Soil + Contaminant + Geometry"""
        fig = plt.figure(figsize=(11, 14))
        gs = gridspec.GridSpec(5, 3, height_ratios=[0.8, 0.5, 1, 0.1, 1], 
                              width_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        
        # Section 1: Configuration Parameters (spans all columns)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        config_text = f"""
SIMULATION CONFIGURATION

Simulation Name: {config.name}
Start Date: {config.start_datetime.strftime('%Y-%m-%d %H:%M')}
End Date: {config.end_datetime.strftime('%Y-%m-%d %H:%M')}
Duration: {config.t_end_hours:.1f} hours ({config.t_end/(24*3600):.1f} days)
Time Step: {config.dt:.1f} seconds ({config.dt/3600:.2f} hours)

Domain: {domain.Lx} m × {domain.Ly} m | Mesh: {domain.nx} × {domain.ny}
"""
        ax1.text(0.1, 0.95, config_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Section 2: Contaminant Properties (spans all columns)
        if contaminant_props:
            ax2 = fig.add_subplot(gs[1, :])
            ax2.axis('off')
            
            contam_text = "CONTAMINANT PROPERTIES\n\n"
            for key, value in contaminant_props.items():
                contam_text += f"{key}: {value}\n"
            
            ax2.text(0.1, 0.9, contam_text, transform=ax2.transAxes,
                    fontsize=10, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # Section 3: Material Curves (compact, 3 columns)
        materials = {}
        if hasattr(domain, 'materials'):
            materials = domain.materials
        
        if materials:
            temp_fig = plot_material_curves(materials, figsize=(12, 3))
            curve_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
            
            for i, (ax_src, ax_dst) in enumerate(zip(temp_fig.axes, curve_axes)):
                for line in ax_src.get_lines():
                    ax_dst.plot(line.get_xdata(), line.get_ydata(),
                              label=line.get_label(), linewidth=1.5)
                ax_dst.set_xlabel(ax_src.get_xlabel(), fontweight='bold', fontsize=8)
                ax_dst.set_ylabel(ax_src.get_ylabel(), fontweight='bold', fontsize=8)
                ax_dst.set_title(ax_src.get_title(), fontweight='bold', fontsize=9)
                ax_dst.grid(True, alpha=0.3)
                ax_dst.legend(fontsize=7)
                ax_dst.tick_params(labelsize=7)
                if hasattr(ax_src, 'get_xlim'):
                    ax_dst.set_xlim(ax_src.get_xlim())
                if hasattr(ax_src, 'get_ylim'):
                    ax_dst.set_ylim(ax_src.get_ylim())
            plt.close(temp_fig)
        
        # Section 4: Domain Geometry (spans all columns)
        boundary_conditions = getattr(field_map, 'bc_manager', None)
        water_table = getattr(config, 'water_table_level', None)
        temp_domain_fig = plot_domain_geometry(domain, boundary_conditions=boundary_conditions,
                                             water_table_level=water_table)
        
        ax4 = fig.add_subplot(gs[4, :])
        temp_ax = temp_domain_fig.axes[0]
        
        # Recreate patches (rectangles) - can't copy patches between figures
        import matplotlib.patches as mpatches
        for patch in temp_ax.patches:
            # Create new patch with same properties
            if isinstance(patch, mpatches.Rectangle):
                new_patch = mpatches.Rectangle(
                    patch.get_xy(), patch.get_width(), patch.get_height(),
                    linewidth=patch.get_linewidth(),
                    edgecolor=patch.get_edgecolor(),
                    facecolor=patch.get_facecolor(),
                    hatch=patch.get_hatch(),
                    label=patch.get_label() if patch.get_label() != '_nolegend_' else None
                )
                ax4.add_patch(new_patch)
        
        # Copy lines
        for line in temp_ax.get_lines():
            ax4.plot(line.get_xdata(), line.get_ydata(),
                    color=line.get_color(), linestyle=line.get_linestyle(),
                    linewidth=line.get_linewidth(), 
                    label=line.get_label() if line.get_label() != '_nolegend_' else None)
        
        # Copy text annotations
        for text in temp_ax.texts:
            bbox_props = None
            if text.get_bbox_patch():
                bbox = text.get_bbox_patch()
                bbox_props = dict(boxstyle=bbox.get_boxstyle(), 
                                 facecolor=bbox.get_facecolor(),
                                 edgecolor=bbox.get_edgecolor(),
                                 linewidth=bbox.get_linewidth(),
                                 alpha=bbox.get_alpha())
            
            ax4.text(text.get_position()[0], text.get_position()[1],
                    text.get_text(), ha=text.get_ha(), va=text.get_va(),
                    fontsize=text.get_fontsize()*0.9, fontweight=text.get_fontweight(),
                    bbox=bbox_props)
        
        # Copy properties
        ax4.set_xlabel(temp_ax.get_xlabel(), fontweight='bold', fontsize=10)
        ax4.set_ylabel(temp_ax.get_ylabel(), fontweight='bold', fontsize=10)
        ax4.set_title(temp_ax.get_title(), fontweight='bold', fontsize=11)
        ax4.set_xlim(temp_ax.get_xlim())
        ax4.set_ylim(temp_ax.get_ylim())
        ax4.set_aspect('equal')
        ax4.grid(True, alpha=0.3)
        
        # Recreate legend
        if temp_ax.get_legend():
            handles, labels = ax4.get_legend_handles_labels()
            if handles:
                ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
        
        plt.close(temp_domain_fig)
        
        fig.suptitle('COUPLED FLOW-TRANSPORT SIMULATION - MODEL SETUP',
                    fontsize=16, fontweight='bold', y=0.98)
        
        return fig
    
    def _create_timeseries_page(self, plotter, field_type='water_table', plotting_config: dict = None):
        """Create time series results page using ResultsPlotter"""
        if plotting_config is None:
            plotting_config = {
                'time_series_fields': [field_type],
                'plot_snapshots': False,  # Only time series for this page
                'plot_comsol_comparison': False,
                'plot_measured_comparison': False
            }
        else:
            # Ensure we only plot time series and the correct field
            plotting_config = plotting_config.copy()
            plotting_config['time_series_fields'] = [field_type]
            plotting_config['plot_snapshots'] = False

        # Use plot_complete_results to generate the figure
        fig = plotter.plot_complete_results(
            plotting_config=plotting_config,
            return_figure=True
        )

        if field_type == 'water_table':
            title = 'WATER TABLE ELEVATION - TIME SERIES'
        elif field_type == 'concentration':
            title = 'CONTAMINANT CONCENTRATION - TIME SERIES'
        else:
            title = f'{field_type.upper()} - TIME SERIES'

        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        return fig
    
    def _create_snapshots_page(self, plotter, field_type='saturation', plotting_config: dict = None):
        """Create spatial snapshots page using ResultsPlotter"""
        if plotting_config is None:
            plotting_config = {
                'time_series_fields': [],  # No time series for this page
                'plot_snapshots': True,
                'snapshot_fields': [field_type],
                'plot_comsol_comparison': False,
                'plot_measured_comparison': False
            }
        else:
            # Ensure we only plot snapshots and the correct field
            plotting_config = plotting_config.copy()
            plotting_config['time_series_fields'] = []
            plotting_config['plot_snapshots'] = True
            plotting_config['snapshot_fields'] = plotting_config.get('snapshot_fields', [field_type])

        # Use plot_complete_results to generate the figure
        fig = plotter.plot_complete_results(
            plotting_config=plotting_config,
            return_figure=True
        )

        if fig is None:
            # No snapshots available, create a placeholder
            fig = plt.figure(figsize=(11, 8.5))
            ax = fig.add_subplot(111)
            ax.text(0.5, 0.5, 'No snapshots available\n(snapshot_manager not provided or empty)',
                   ha='center', va='center', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')

        if field_type == 'saturation':
            title = 'SOIL SATURATION - SPATIAL DISTRIBUTION'
        elif field_type == 'concentration':
            title = 'CONTAMINANT CONCENTRATION - SPATIAL DISTRIBUTION'
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        return fig

    def print_transport_verification_report(self, config, domain, plotter, 
                                          report_config, filename=None):
        """
        Generate transport verification report with custom pages
        
        Parameters:
        -----------
        config : SimulationConfig
            Simulation configuration
        domain : Domain  
            Domain object
        plotter : ResultsPlotter
            Results plotter with simulation data
        report_config : dict
            Configuration with custom page definitions
        filename : str, optional
            Output filename
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transport_verification_report_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        
        print(f"\nGenerating transport verification report...")
        print(f"Output: {filepath}")
        
        with PdfPages(str(filepath)) as pdf:
            # Page 1: Parameters (if requested)
            if report_config.get('parameters', False):
                print("  - Parameters page")
                fig = self._create_parameters_page(config, domain)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 2: Model setup (if requested) 
            if report_config.get('model_setup', False):
                print("  - Model setup page")
                fig = self._create_model_setup_page(domain)
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 3: Standard timeseries (if requested)
            if report_config.get('time_series_plots', False):
                print("  - Standard timeseries page")
                fig = self._create_timeseries_page(plotter, 'concentration')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Page 4: Standard snapshots (if requested)
            if report_config.get('snapshot_plots', False):
                print("  - Standard snapshots page")
                fig = self._create_snapshots_page(plotter, 'concentration')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # Custom pages
            for key, page_config in report_config.items():
                if not key.startswith('custom_page_'):
                    continue
                    
                page_name = key.replace('custom_page_', '').replace('_', ' ').title()
                print(f"  - Custom page: {page_name}")
                
                if page_config['type'] == 'timeseries_comparison':
                    fig = self._create_comparison_timeseries_page(page_config)
                elif page_config['type'] == 'timeseries':
                    fig = self._create_custom_timeseries_page(page_config)
                elif page_config['type'] == 'snapshots':
                    fig = self._create_custom_snapshots_page(page_config)
                else:
                    print(f"    Warning: Unknown page type '{page_config['type']}'")
                    continue
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
        
        print(f"Report generated: {filepath}")
        return filepath

    def _create_comparison_timeseries_page(self, page_config):
        """Create comparison timeseries page (simulated vs analytical)"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        sim_data = page_config['sim_data']
        analytical_data = page_config['analytical_data']
        
        # Convert times to hours
        times_hours = np.array(sim_data['times']) / 3600.0
        probe_names = [k for k in sim_data.keys() if k != 'times']
        colors = plt.rcParams.get('axes.prop_cycle').by_key()['color']
        
        for i, probe_name in enumerate(probe_names):
            color = colors[i % len(colors)]
            
            # Simulated
            ax.plot(times_hours, sim_data[probe_name], 
                   color=color, linestyle='-', linewidth=2.5, 
                   label=f'{probe_name} (Simulated)', alpha=0.8)
            
            # Analytical
            ax.plot(times_hours, analytical_data[probe_name], 
                   color=color, linestyle='--', linewidth=2, 
                   label=f'{probe_name} (Analytical)', alpha=0.8)
        
        ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax.set_ylabel(page_config.get('ylabel', 'Value'), fontweight='bold', fontsize=12)
        ax.set_title(page_config['title'], fontweight='bold', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if len(times_hours) > 0:
            ax.set_xlim(times_hours[0], times_hours[-1])
        
        plt.tight_layout()
        return fig

    def _create_custom_timeseries_page(self, page_config):
        """Create custom timeseries page"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        data = page_config['data']
        times_hours = np.array(data['times']) / 3600.0
        probe_names = [k for k in data.keys() if k != 'times']
        colors = plt.rcParams.get('axes.prop_cycle').by_key()['color']
        
        for i, probe_name in enumerate(probe_names):
            color = colors[i % len(colors)]
            ax.plot(times_hours, data[probe_name], 
                   color=color, linestyle='-', linewidth=2, 
                   label=probe_name, alpha=0.8)
        
        ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax.set_ylabel(page_config.get('ylabel', 'Value'), fontweight='bold', fontsize=12)
        ax.set_title(page_config['title'], fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if len(times_hours) > 0:
            ax.set_xlim(times_hours[0], times_hours[-1])
        
        plt.tight_layout()
        return fig

    def _create_custom_snapshots_page(self, page_config):
        """Create custom snapshots page"""
        snapshots = page_config['data']
        snapshot_times = list(snapshots.keys())
        n_snapshots = len(snapshot_times)
        
        if n_snapshots == 0:
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.text(0.5, 0.5, 'No snapshots available', ha='center', va='center',
                   fontsize=14, fontweight='bold')
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            ax.set_title(page_config['title'], fontweight='bold', fontsize=16)
            return fig
        
        # Arrange subplots
        cols = min(3, n_snapshots)
        rows = (n_snapshots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(11, 8.5))
        if n_snapshots == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        from firedrake import plot
        
        for i, t in enumerate(snapshot_times):
            if i < len(axes):
                try:
                    plot(snapshots[t], axes=axes[i])
                    axes[i].set_title(f't = {t/3600:.1f}h', fontweight='bold')
                    axes[i].set_aspect('equal')
                except Exception as e:
                    axes[i].text(0.5, 0.5, f'Plot error:\n{str(e)}', 
                               ha='center', va='center', fontsize=10)
                    axes[i].set_xlim(0, 1)
                    axes[i].set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(n_snapshots, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(page_config['title'], fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
