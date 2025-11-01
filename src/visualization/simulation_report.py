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
from .basic_plotting import BasicPlotting


class SimulationReport(BasicPlotting):
    
    def __init__(self, output_dir=None):
        self.output_dir = Path(output_dir) if output_dir else Path("results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    # ============================================================================
    # PUBLIC API - Main report generators
    # ============================================================================
    
    def print_richards_report(self, config, domain, plotter, boundary_conditions=None, filename=None):
        """Generate Richards equation report: setup + timeseries + snapshots"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"richards_report_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        print(f"\n📄 Generating Richards report: {filepath}")
        
        with PdfPages(str(filepath)) as pdf:
            # Page 1: Configuration + Materials + Domain geometry
            print("  - Setup page")
            fig = self._create_richards_setup_page(config, domain, boundary_conditions)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Water table time series
            print("  - Time series page")
            fig = self._create_timeseries_page(plotter, 'water_table')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 3: Saturation snapshots
            if plotter.snapshot_manager:
                print("  - Snapshots page")
                fig = self._create_snapshots_page(plotter, 'saturation')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # PDF metadata
            d = pdf.infodict()
            d['Title'] = 'Richards Equation Simulation Report'
            d['Author'] = 'Firedrake Simulation'
            d['Subject'] = f"Simulation: {config.name}"
            d['CreationDate'] = datetime.now()
        
        print(f"✓ Report saved: {filepath}")
        return filepath
    
    def print_transport_report(self, config, domain, plotter, contaminant_props=None, filename=None):
        """Generate transport report: setup + timeseries + snapshots"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transport_report_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        print(f"\n📄 Generating Transport report: {filepath}")
        
        with PdfPages(str(filepath)) as pdf:
            # Page 1: Configuration + Contaminant + Materials + Domain
            print("  - Setup page")
            fig = self._create_transport_setup_page(config, domain, contaminant_props)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Concentration time series
            print("  - Time series page")
            fig = self._create_timeseries_page(plotter, 'concentration')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 3: Concentration snapshots
            if plotter.snapshot_manager:
                print("  - Snapshots page")
                fig = self._create_snapshots_page(plotter, 'concentration')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # PDF metadata
            d = pdf.infodict()
            d['Title'] = 'Coupled Flow-Transport Simulation Report'
            d['Author'] = 'Firedrake Simulation'
            d['Subject'] = f"Simulation: {config.name}"
            d['CreationDate'] = datetime.now()
        
        print(f"✓ Report saved: {filepath}")
        return filepath
    
    def print_custom_report(self, config, domain, plotter, custom_pages, filename=None):
        """
        Generate custom report with user-defined pages
        
        custom_pages format:
        {
            'comparison': {
                'type': 'timeseries_comparison',
                'sim_data': {...},
                'analytical_data': {...},
                'title': 'Simulated vs Analytical',
                'ylabel': 'Concentration (kg/m³)'
            },
            'residuals': {
                'type': 'timeseries',
                'data': {...},
                'title': 'Residuals',
                'ylabel': 'Residual (kg/m³)'
            },
            'snapshots': {
                'type': 'snapshots',
                'data': {...},
                'title': 'Concentration Field'
            }
        }
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_report_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        print(f"\n📄 Generating custom report: {filepath}")
        
        with PdfPages(str(filepath)) as pdf:
            for page_name, page_config in custom_pages.items():
                print(f"  - {page_name}")
                
                page_type = page_config.get('type')
                
                # Route to appropriate page creator based on type
                if page_type == 'timeseries_comparison':
                    fig = self._create_comparison_timeseries_page(page_config)
                elif page_type == 'timeseries':
                    fig = self._create_custom_timeseries_page(page_config)
                elif page_type == 'snapshots':
                    fig = self._create_custom_snapshots_page(page_config)
                else:
                    print(f"    Warning: Unknown page type '{page_type}'")
                    continue
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # PDF metadata
            d = pdf.infodict()
            d['Title'] = 'Custom Simulation Report'
            d['Author'] = 'Firedrake Simulation'
            d['Subject'] = f"Simulation: {config.name}"
            d['CreationDate'] = datetime.now()
        
        print(f"✓ Report saved: {filepath}")
        return filepath
    
    # ============================================================================
    # STANDARD PAGE CREATORS - Used by main reports
    # ============================================================================
    
    def _create_richards_setup_page(self, config, domain, boundary_conditions):
        """Create setup page: config text + material curves + domain geometry"""
        fig = plt.figure(figsize=(11, 14))
        gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1.2, 0.1, 1],
                              width_ratios=[1, 1, 1, 1], hspace=0.35, wspace=0.25)
        
        # Top: Configuration text box
        self._add_config_text(fig, gs[0, :], config, domain)
        
        # Middle: Material hydraulic curves (4 subplots)
        if hasattr(domain, 'materials'):
            self._add_material_curves(fig, gs, row=1, materials=domain.materials)
        
        # Bottom: Domain geometry with regions
        self._add_domain_geometry(fig, gs, row=3, domain=domain, boundary_conditions=boundary_conditions)
        
        fig.suptitle('RICHARDS EQUATION SIMULATION - MODEL SETUP',
                    fontsize=16, fontweight='bold', y=0.98)
        return fig
    
    def _create_transport_setup_page(self, config, domain, contaminant_props):
        """Create setup page: config + contaminant + materials + geometry"""
        fig = plt.figure(figsize=(11, 14))
        gs = gridspec.GridSpec(5, 3, height_ratios=[0.8, 0.5, 1, 0.1, 1],
                              width_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        
        # Top: Configuration text
        self._add_config_text(fig, gs[0, :], config, domain)
        
        # Contaminant properties text
        if contaminant_props:
            self._add_contaminant_text(fig, gs[1, :], contaminant_props)
        
        # Material curves (3 subplots - more compact)
        if hasattr(domain, 'materials'):
            self._add_material_curves(fig, gs, row=2, materials=domain.materials, n_plots=3)
        
        # Domain geometry
        self._add_domain_geometry(fig, gs, row=4, domain=domain)
        
        fig.suptitle('COUPLED FLOW-TRANSPORT SIMULATION - MODEL SETUP',
                    fontsize=16, fontweight='bold', y=0.98)
        return fig
    
    def _create_timeseries_page(self, plotter, field_type):
        """Create time series page using ResultsPlotter"""
        plotting_config = {
            'time_series_fields': [field_type],
            'plot_snapshots': False,
            'plot_comsol_comparison': True,
            'plot_measured_comparison': True
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
    
    # ============================================================================
    # CUSTOM PAGE CREATORS - For verification/comparison reports
    # ============================================================================
    
    def _create_comparison_timeseries_page(self, config):
        """Plot simulated vs analytical time series on same axes"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        sim_data = config['sim_data']
        analytical_data = config['analytical_data']
        times_hours = np.array(sim_data['times']) / 3600.0
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Plot each probe: solid line = simulated, dashed = analytical
        for i, probe_name in enumerate([k for k in sim_data.keys() if k != 'times']):
            color = colors[i % len(colors)]
            ax.plot(times_hours, sim_data[probe_name], color=color, linestyle='-',
                   linewidth=2.5, label=f'{probe_name} (Simulated)', alpha=0.8)
            ax.plot(times_hours, analytical_data[probe_name], color=color, linestyle='--',
                   linewidth=2, label=f'{probe_name} (Analytical)', alpha=0.8)
        
        ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax.set_ylabel(config.get('ylabel', 'Value'), fontweight='bold', fontsize=12)
        ax.set_title(config['title'], fontweight='bold', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if len(times_hours) > 0:
            ax.set_xlim(times_hours[0], times_hours[-1])
        
        plt.tight_layout()
        return fig
    
    def _create_custom_timeseries_page(self, config):
        """Plot custom time series data (e.g., residuals)"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        data = config['data']
        times_hours = np.array(data['times']) / 3600.0
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Plot each series
        for i, probe_name in enumerate([k for k in data.keys() if k != 'times']):
            ax.plot(times_hours, data[probe_name], color=colors[i % len(colors)],
                   linestyle='-', linewidth=2, label=probe_name, alpha=0.8)
        
        ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax.set_ylabel(config.get('ylabel', 'Value'), fontweight='bold', fontsize=12)
        ax.set_title(config['title'], fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if len(times_hours) > 0:
            ax.set_xlim(times_hours[0], times_hours[-1])
        
        plt.tight_layout()
        return fig
    
    def _create_custom_snapshots_page(self, config):
        """Plot custom spatial snapshots"""
        from firedrake import tripcolor
        
        snapshots = config['data']
        snapshot_times = sorted(list(snapshots.keys()))
        n_snapshots = len(snapshot_times)
        
        if n_snapshots == 0:
            return self._create_no_snapshots_placeholder(config['title'])
        
        # Arrange in grid (max 3 columns)
        cols = min(3, n_snapshots)
        rows = (n_snapshots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(11, 8.5))
        if n_snapshots == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        # Plot each snapshot
        for i, t in enumerate(snapshot_times):
            if i < len(axes):
                try:
                    tripcolor(snapshots[t]['concentration'], axes=axes[i])
                    axes[i].set_title(f't = {t/3600:.1f}h', fontweight='bold')
                    axes[i].set_aspect('equal')
                except Exception:
                    axes[i].text(0.5, 0.5, 'Plot error', ha='center', va='center')
                    axes[i].set_xlim(0, 1)
                    axes[i].set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(n_snapshots, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(config['title'], fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    def _create_no_snapshots_placeholder(self, title='SPATIAL DISTRIBUTION'):
        """Placeholder when no snapshots available"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.text(0.5, 0.5, 'No snapshots available', ha='center', va='center',
               fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        fig.suptitle(title, fontsize=16, fontweight='bold')
        return fig
    
    # ============================================================================
    # SECTION BUILDERS - Add specific content to setup pages
    # ============================================================================
    
    def _add_config_text(self, fig, gridspec_slice, config, domain):
        """Add configuration text box to page"""
        ax = fig.add_subplot(gridspec_slice)
        ax.axis('off')
        
        config_text = f"""
SIMULATION CONFIGURATION

Simulation Name: {config.name}
Start: {config.start_datetime.strftime('%Y-%m-%d %H:%M')}
End: {config.end_datetime.strftime('%Y-%m-%d %H:%M')}
Duration: {config.t_end_hours:.1f} hours ({config.t_end/(24*3600):.1f} days)
Time Step: {config.dt:.1f} seconds

Domain: {domain.Lx} m × {domain.Ly} m | Mesh: {domain.nx} × {domain.ny}
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
    
    def _add_material_curves(self, fig, gs, row, materials, n_plots=4):
        """
        Add material hydraulic curves to page
        Creates temporary figure with curves, then copies to main figure
        """
        # Create curves in temporary figure
        temp_fig = self.plot_material_curves(materials)
        
        # Create subplots in main figure and copy content
        for i in range(n_plots):
            ax_dst = fig.add_subplot(gs[row, i])
            ax_src = temp_fig.axes[i]
            self._copy_axis(ax_src, ax_dst)
        
        plt.close(temp_fig)
    
    def _add_domain_geometry(self, fig, gs, row, domain, boundary_conditions=None):
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
        
        # Create geometry in temporary figure
        temp_fig, temp_ax = self.plot_domain_geometry(domain.Lx, domain.Ly, regions)
        
        # Add water table if available
        if boundary_conditions:
            self.plot_water_table(temp_ax, t=0.0, bc_manager=boundary_conditions, domain=domain)
        
        # Copy to main figure
        ax = fig.add_subplot(gs[row, :])
        self._copy_axis(temp_ax, ax)
        
        plt.close(temp_fig)
    
    # ============================================================================
    # UTILITY - Copy matplotlib axes between figures
    # ============================================================================
    
    def _copy_axis(self, ax_src, ax_dst):
        """
        Copy all content from one axis to another
        Needed because matplotlib doesn't allow moving axes between figures
        Copies: patches, lines, text, labels, limits, aspect, grid, legend
        """
        import matplotlib.patches as mpatches
        
        # Copy patches (rectangles for domain regions)
        for patch in ax_src.patches:
            if isinstance(patch, mpatches.Rectangle):
                new_patch = mpatches.Rectangle(
                    patch.get_xy(), patch.get_width(), patch.get_height(),
                    linewidth=patch.get_linewidth(), edgecolor=patch.get_edgecolor(),
                    facecolor=patch.get_facecolor(), hatch=patch.get_hatch(),
                    label=patch.get_label() if patch.get_label() != '_nolegend_' else None
                )
                ax_dst.add_patch(new_patch)
        
        # Copy lines (material curves, water table)
        for line in ax_src.get_lines():
            ax_dst.plot(line.get_xdata(), line.get_ydata(), color=line.get_color(),
                    linestyle=line.get_linestyle(), linewidth=line.get_linewidth(),
                    label=line.get_label() if line.get_label() != '_nolegend_' else None)
        
        # Copy text annotations
        for text in ax_src.texts:
            bbox_props = None
            if text.get_bbox_patch():
                bbox = text.get_bbox_patch()
                bbox_props = dict(boxstyle=bbox.get_boxstyle(), facecolor=bbox.get_facecolor(),
                                edgecolor=bbox.get_edgecolor(), linewidth=bbox.get_linewidth(),
                                alpha=bbox.get_alpha())
            
            # Determine if text uses axis transform or data transform
            transform = ax_dst.transAxes if text.get_transform() == ax_src.transAxes else None
            
            ax_dst.text(text.get_position()[0], text.get_position()[1], text.get_text(),
                    ha=text.get_ha(), va=text.get_va(), fontsize=text.get_fontsize(),
                    fontfamily=text.get_fontfamily(), fontweight=text.get_fontweight(),
                    bbox=bbox_props, transform=transform)
        
        # Copy axis properties
        ax_dst.set_xlabel(ax_src.get_xlabel(), fontweight='bold', fontsize=11)
        ax_dst.set_ylabel(ax_src.get_ylabel(), fontweight='bold', fontsize=11)
        ax_dst.set_title(ax_src.get_title(), fontweight='bold', fontsize=12)
        
        # Copy axis limits and aspect
        if hasattr(ax_src, 'get_xlim'):
            ax_dst.set_xlim(ax_src.get_xlim())
        if hasattr(ax_src, 'get_ylim'):
            ax_dst.set_ylim(ax_src.get_ylim())
        if ax_src.get_aspect() != 'auto':
            ax_dst.set_aspect(ax_src.get_aspect())
        
        # Always enable grid (standard for our reports)
        ax_dst.grid(True, alpha=0.3)
        
        # Copy legend
        if ax_src.get_legend():
            handles, labels = ax_dst.get_legend_handles_labels()
            if handles:
                ax_dst.legend(loc='upper right', fontsize=10, framealpha=0.9)