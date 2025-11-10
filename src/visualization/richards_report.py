from .simulation_report import SimulationReport
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

class RichardsReport(SimulationReport):
    def __init__(self, plotter, output_dir=None):
        super().__init__(plotter, output_dir)

    def print(self, filename="richards_report.pdf"):
        """Generate a Richards-specific simulation report.

        Args:
            plotter (ResultsPlotter): Plotter with simulation results.
            filename (str, optional): Output PDF filename. Defaults to "richards_report.pdf".
        """
        figures = {}
        
        # Page 1: Model Setup
        print("  - Model setup page")
        fig = plt.figure(figsize=(11, 14))
        gs = gridspec.GridSpec(4, 4, height_ratios=[1, 1.2, 0.1, 1], figure=fig,
                              width_ratios=[1, 1, 1, 1], hspace=0.35, wspace=0.25)
        ax_text = fig.add_subplot(gs[0, :])
        ax_curves = [fig.add_subplot(gs[1, i]) for i in range(4)]
        ax_geometry = fig.add_subplot(gs[2:4, :])

        # Top: Configuration text box
        self._add_config_text(self.config, self.domain, ax_text)
        self._add_material_curves(self.domain.materials, ax_curves)
        self._add_domain_geometry(self.domain, self.bc_manager, ax_geometry)
        
        fig.suptitle('RICHARDS EQUATION SIMULATION - MODEL SETUP',
                    fontsize=16, fontweight='bold', y=0.98)
        
        figures['Setup'] = fig
        
        # Page 2: Water table time series
        print("  - Time series page")
        fig = self._create_timeseries_page(self.plotter, 'water_table')
        figures['Time Series'] = fig
        
        # Page 3: Saturation snapshots
        if self.plotter.snapshot_manager:
            print("  - Snapshots page")
            fig = self._create_snapshots_page(self.plotter, 'saturation')
            figures['Snapshots'] = fig
        
        # Save PDF with metadata
        metadata = {
            'Title': 'Richards Equation Simulation Report',
            'Author': 'FireDrake Simulation',
            'Subject': f"Simulation: {self.config.project_name}",
            'Creator': 'FireDrake Framework'
        }
        
        self.save_pdf(figures, filename, metadata)
        
        # Close all figures to free memory
        for fig in figures.values():
            plt.close(fig)
        
        return self.output_dir / filename
        
