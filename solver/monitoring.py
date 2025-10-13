"""
Monitoring utilities for time series and spatial snapshots
"""
import numpy as np

class ProbeManager:
    """Manages pressure monitoring at specific points"""
    
    def __init__(self, points):
        """
        Initialize probe manager
        
        Args:
            points: List of (x, y, name) tuples
        """
        self.points = points
        self.data = {name: [] for _, _, name in points}
        self.times = []
    
    def record(self, t, pressure_function):
        """Record pressure at all probe locations"""
        self.times.append(t)
        for x, y, name in self.points:
            try:
                p_val = pressure_function.at([x, y])
                # Water table elevation = sensor elevation + pressure head
                # If p > 0: water table is above sensor (saturated/ponded)
                # If p < 0: water table is below sensor (unsaturated)
                # If p = 0: water table is exactly at sensor elevation
                water_table_elevation = y + p_val
                self.data[name].append(water_table_elevation)
            except:
                self.data[name].append(float('nan'))
    
    def get_data(self):
        """
        Get monitoring data
        
        Returns:
            Dictionary with 'times' and 'data' keys
        """
        return {
            'times': np.array(self.times),
            'data': self.data
        }
    
    def get_time_hours(self):
        """Get times in hours"""
        return np.array(self.times) / 3600.0


class SnapshotManager:
    """Manages spatial snapshots at specific times"""
    
    def __init__(self, snapshot_times: list, domain):
        """
        Initialize snapshot manager
        
        Args:
            snapshot_times: List of times (in seconds) to save snapshots
            domain: Domain object for computing saturation
        """
        self.snapshot_times = snapshot_times
        self.snapshots = {}
        self.domain = domain
    
    def should_record(self, t: float, dt: float) -> bool:
        """
        Check if current time is close to a snapshot time
        
        Args:
            t: Current time
            dt: Time step size
        
        Returns:
            True if should record snapshot
        """
        return any(abs(t - st) < dt/2 for st in self.snapshot_times)
    
    def record(self, t: float, pressure_field):
        """
        Record snapshot at current time
        
        Args:
            t: Current time
            pressure_field: Firedrake Function with pressure head
        """
        # Compute saturation from pressure
        saturation = self.domain.compute_saturation_field(pressure_field)
        
        # Store deep copies
        self.snapshots[t] = {
            'pressure': pressure_field.copy(deepcopy=True),
            'saturation': saturation.copy(deepcopy=True)
        }
        
        # Print diagnostic info
        sat_vals = saturation.dat.data[:]
        p_vals = pressure_field.dat.data[:]
        print(f"  Saved snapshot at t={t/3600:.2f}h")
        print(f"    Saturation range: [{sat_vals.min():.4f}, {sat_vals.max():.4f}]")
        print(f"    Pressure range: [{p_vals.min():.4f}, {p_vals.max():.4f}]")
    
    def get_sorted_times(self):
        """Get sorted list of snapshot times"""
        return sorted(self.snapshots.keys())