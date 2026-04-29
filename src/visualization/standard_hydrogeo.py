from .report_base import ReportBase
from .basic_plotting import * 
from .standard_page_01 import *
from pathlib import Path

class HydrogeoSimulationReport(ReportBase):
    def __init__(self, solver, filename):
        super().__init__(filename)
        self.config = solver.config
        self.domain = solver.domain 
        self.probes = solver.probe_manager
        self.scenario = solver.source_scenario

    def build(self):
        """Orchestrate the report structure here."""
        self._page_model_setup()
        self._page_timeseries()
        #self._page_images()
    
    @classmethod
    def print(cls, solver, filename="results"):
        filename = Path(solver.config.output_dir) / filename
        with cls(solver, filename) as report:
            report.build()

    def _page_model_setup(self):
        fig, gs = self.create_page(rows=4, cols=2)
        ax_config = fig.add_subplot(gs[0,:])
        ax_config.axis("off")
        add_config_text(ax_config, self.config, self.domain)

        ax_domain = fig.add_subplot(gs[1,:])
        plot_domain(ax_domain, self.domain)

        ax_mat_params = fig.add_subplot(gs[2, 0])
        plot_material_report(ax_mat_params, self.domain.materials)

        ax_theta = fig.add_subplot(gs[3, 0])
        ax_kr = fig.add_subplot(gs[3, 1])
        plot_material_curves(ax_theta, ax_kr, self.domain.materials)

    def _page_timeseries(self):
        fig, gs = self.create_page(rows=1, cols=1, landscape=True)
        ax = fig.add_subplot(gs[:,:])
        plot_timeseries(ax, self.probes.get_data(), 'water_table', self.scenario.events)

    def _page_images(self):
        fig, gs = self.create_page(rows=1, cols=1, landscape=True)
        ax = fig.add_subplot(gs[:,:])
        


