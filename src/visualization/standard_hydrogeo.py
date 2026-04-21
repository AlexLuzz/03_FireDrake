from .report_base import ReportBase
from .basic_plotting import * 
from .standard_pages import *

class HydrogeoSimulationReport(ReportBase):
    def __init__(self, filename, config, domain, probe_manager):
        super().__init__(filename)
        self.config = config
        self.domain = domain 
        self.probes = probe_manager

    def build(self):
        """Orchestrate the report structure here."""
        self._page_model_setup()
        self._page_timeseries_results()
        #self._page_spatial_snapshots()

    def _page_model_setup(self):
        fig, gs = self.create_page(rows=2, cols=1)
        ax_config = fig.add_subplot(gs[0,:])
        ax_domain = fig.add_subplot(gs[1,:])
        add_config_text(ax_config, self.config, self.domain)
        plot_domain(ax_domain, self.domain)
        #plot_material_curves(axes[1], self.data.materials)


    def _page_timeseries_results(self):
        fig, gs = self.create_page(rows=1, cols=1)
        ax = fig.add_subplot(gs[:,:])
        times = self.probes.times
        for probe in self.probes.data.keys():
            ax.plot(times, self.probes.get_probe_data(probe, "water_table"))


