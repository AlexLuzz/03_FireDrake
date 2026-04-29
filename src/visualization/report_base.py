from abc import ABC, abstractmethod
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from contextlib import contextmanager

class ReportBase(ABC):
    def __init__(self, filename="results.pdf"):
        self.filename = filename
        self.pdf = None
        self._current_figs = []

    def __enter__(self):
        self.pdf = PdfPages(self.filename)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for fig in self._current_figs:
            self.pdf.savefig(fig)
            plt.close(fig)

        self.pdf.close()
        print(f"✓ Report saved: {self.filename}")

    def create_page(
        self,
        rows=1,
        cols=1,
        landscape=False,
        height_ratios=None,
        width_ratios=None,
        margins=None,
    ):
        figsize = (11.69, 8.27) if landscape else (8.27, 11.69)

        fig = plt.figure(figsize=figsize)

        margins = margins or dict(left=0.08, right=0.95, top=0.95, bottom=0.07)

        gs = fig.add_gridspec(
            nrows=rows,
            ncols=cols,
            height_ratios=height_ratios,
            width_ratios=width_ratios,
            hspace=0.3,
            wspace=0.2,
            **margins
        )

        # track figure for later saving
        self._current_figs.append(fig)

        return fig, gs
    
    @contextmanager
    def page(self, *args, **kwargs):
        fig, gs = self.create_page(*args, **kwargs)
        try:
            yield fig, gs
            self.pdf.savefig(fig)
        finally:
            plt.close(fig)

    @abstractmethod
    def build(self):
        """This is where you define the sequence of pages."""
        pass