"""
Setup module: Configuration, rain scenarios, and monitoring
"""

from .config import SimulationConfig
from .source_term import SourceManager
from .monitoring import SimulationMonitor

__all__ = [
    'SimulationConfig',
    'SourceManager',
    'SimulationMonitor',
    'rainfall_scenario',
]
