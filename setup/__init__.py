"""
Setup module: Configuration, rain scenarios, and monitoring
"""

from .config import SimulationConfig
from .source_term import SourceScenario, Zone
from .monitoring import ProbeManager, SnapshotManager

__all__ = [
    'SimulationConfig',
    'SourceScenario',
    'Zone',
    'ProbeManager',
    'SnapshotManager',
]
