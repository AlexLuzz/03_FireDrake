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
                            filename=None, include_snapshots=True):
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
            self._create_richards_setup_page(config, domain, field_map)
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # PAGE 2: Time Series Results
            self._create_timeseries_page(plotter, field_type='water_table')
            pdf.savefig(bbox_inches='tight')
            plt.close()
            
            # PAGE 3: Spatial Snapshots (if requested and available)
            if include_snapshots and plotter.snapshot_manager:
                self._create_snapshots_page(plotter, field_type='saturation')
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
    
    def _create_richards_setup_page(self, config, domain, field_map):
        """Create page 1 for Richards report: Config + Soil + Geometry"""
        fig = plt.figure(figsize=(11, 14))
        gs = gridspec.GridSpec(4, 3, height_ratios=[1, 1.2, 0.1, 1], 
                              width_ratios=[1, 1, 1], hspace=0.35, wspace=0.3)
        
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
        
        # Section 2: Material Curves (3 columns)
        # Get materials from domain
        materials = {}
        if hasattr(domain, 'materials'):
            materials = domain.materials
        
        # Plot material curves directly in the gridspec
        if materials:
            # Create the curves plot and extract data
            temp_fig = plot_material_curves(materials)
            
            # Create 3 subplots in row 1 (material curves)
            curve_axes = [fig.add_subplot(gs[1, i]) for i in range(3)]
            
            # Copy the curves to the main figure
            for i, (ax_src, ax_dst) in enumerate(zip(temp_fig.axes, curve_axes)):
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
            
            plt.close(temp_fig)
        
        # Section 3: Domain Geometry (spans all columns, bottom section)
        # Get water table level if available
        water_table = getattr(config, 'water_table_level', None)
        if water_table is None and hasattr(field_map, 'bc_manager'):
            # Try to get from boundary conditions
            bc = field_map.bc_manager
            if hasattr(bc, 'left_wt'):
                water_table = bc.left_wt
        
        # Plot domain geometry in a temporary figure and extract info
        temp_domain_fig = plot_domain_geometry(domain, water_table_level=water_table)
        
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
        water_table = getattr(config, 'water_table_level', None)
        temp_domain_fig = plot_domain_geometry(domain, water_table_level=water_table)
        
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
    
    def _create_timeseries_page(self, plotter, field_type='water_table'):
        """Create time series results page"""
        # This will use the existing plotter functionality
        # Just call plotter's internal time series method
        fig = plt.figure(figsize=(11, 8.5))
        
        if field_type == 'water_table':
            title = 'WATER TABLE ELEVATION - TIME SERIES'
        elif field_type == 'concentration':
            title = 'CONTAMINANT CONCENTRATION - TIME SERIES'
        else:
            title = f'{field_type.upper()} - TIME SERIES'
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Use plotter's existing time series plotting
        # This is a placeholder - actual implementation would call plotter methods
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Time series plot for {field_type}\n(Use existing plotter functionality)',
               ha='center', va='center', fontsize=12)
        
        return fig
    
    def _create_snapshots_page(self, plotter, field_type='saturation'):
        """Create spatial snapshots page"""
        fig = plt.figure(figsize=(11, 8.5))
        
        if field_type == 'saturation':
            title = 'SOIL SATURATION - SPATIAL DISTRIBUTION'
        elif field_type == 'concentration':
            title = 'CONTAMINANT CONCENTRATION - SPATIAL DISTRIBUTION'
        else:
            title = f'{field_type.upper()} - SPATIAL DISTRIBUTION'
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Use plotter's existing snapshot plotting
        # This is a placeholder
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, f'Snapshot plots for {field_type}\n(Use existing plotter functionality)',
               ha='center', va='center', fontsize=12)
        
        return fig
