# visualization/pdf_generator.py
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Callable, Optional

class SimulationReport:
    """Simple PDF report generator"""
    
    def __init__(self, filename: str = 'report.pdf', title: str = 'Simulation Report'):
        self.filename = filename
        self.title = title
        self._pdf = None
    
    def __enter__(self):
        self._pdf = PdfPages(self.filename)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._pdf:
            # Add metadata
            d = self._pdf.infodict()
            d['Title'] = self.title
            d['Author'] = 'Richards Solver'
            d['CreationDate'] = datetime.now()
            self._pdf.close()
    
    def add_page(self, fig):
        """Add any matplotlib figure as a page"""
        self._pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
    
    def page_params(self, params: Dict, title: str = "Parameters", 
                    ncols: int = 2, figsize: tuple = (8.5, 11)):
        """Create page with parameter table"""
        fig, ax = plt.subplots(figsize=figsize)
        ax.axis('off')
        
        # Title
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Format parameters
        param_list = [[k, str(v)] for k, v in params.items()]
        
        # Split into columns
        n_per_col = len(param_list) // ncols + (len(param_list) % ncols > 0)
        
        for col in range(ncols):
            start = col * n_per_col
            end = min(start + n_per_col, len(param_list))
            if start >= len(param_list):
                break
            
            col_data = param_list[start:end]
            x_pos = col / ncols + 0.05
            
            for i, (key, val) in enumerate(col_data):
                y_pos = 0.90 - i * 0.03
                ax.text(x_pos, y_pos, f"{key}:", fontweight='bold', 
                       transform=fig.transFigure, fontsize=10)
                ax.text(x_pos + 0.25, y_pos, val, 
                       transform=fig.transFigure, fontsize=10)
        
        self.add_page(fig)
    
    def page_figure(self, plot_func: Callable, title: Optional[str] = None, 
                    figsize: tuple = (10, 8), **kwargs):
        """Create page with single figure from plotting function"""
        fig, ax = plt.subplots(figsize=figsize)
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        
        plot_func(ax=ax, **kwargs)
        self.add_page(fig)
    
    def page_grid(self, plot_funcs: List[Callable], titles: Optional[List[str]] = None,
                  nrows: int = 2, ncols: int = 2, figsize: tuple = (11, 8.5), **kwargs):
        """Create page with grid of subplots"""
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
        axes = axes.flatten() if nrows * ncols > 1 else [axes]
        
        for i, (ax, plot_func) in enumerate(zip(axes, plot_funcs)):
            plot_func(ax=ax, **kwargs)
            if titles and i < len(titles):
                ax.set_title(titles[i], fontweight='bold')
        
        # Hide unused axes
        for ax in axes[len(plot_funcs):]:
            ax.axis('off')
        
        plt.tight_layout()
        self.add_page(fig)


# Usage examples
"""
from .pdf_generator import SimulationReport
from .plots import plot_pressure_profile, plot_timeseries, plot_theta_curve

# Simple usage
with SimulationReport('output.pdf', title='My Simulation') as report:
    
    # Page 1: Parameters
    report.page_params({
        'Ks': '1e-5 m/s',
        'porosity': 0.4,
        'soil': 'till',
        'dt': '60 s',
        'simulation_time': '24 h'
    }, title="Configuration")
    
    # Page 2: Single time series
    report.page_figure(
        plot_timeseries, 
        title="Water Table Evolution",
        data=results.water_table_history,
        xlabel='Time [h]',
        ylabel='Depth [m]'
    )
    
    # Page 3: Grid of profiles at different times
    report.page_grid(
        [lambda ax, t=t: plot_pressure_profile(ax, results, t) 
         for t in [0, 6, 12, 24]],
        titles=[f't = {t}h' for t in [0, 6, 12, 24]],
        nrows=2, ncols=2
    )
    
    # Page 4: Custom figure
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title('Custom Plot')
    report.add_page(fig)

# Even simpler for single page
with SimulationReport('quick_report.pdf') as report:
    report.page_params({'result': 'success', 'runtime': '10.5s'})
"""