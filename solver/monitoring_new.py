"""
Flexible monitoring system for multiple field types
"""
import numpy as np


class ProbeManager:
    """Monitor any field type at specific locations"""
    
    def __init__(self, mesh, x_positions, names=None, reference_y=2.5):
        self.mesh = mesh
        self.x_positions = x_positions
        self.names = names or [f"Probe_{i+1}" for i in range(len(x_positions))]
        self.reference_y = reference_y  # y-coordinate where to sample fields
        self.times = []
        self.fields = {}  # {field_name: {probe_name: [values]}}
        
        coords = mesh.coordinates.dat.data
        self.x_tol = (coords[:, 0].max() - coords[:, 0].min()) / (len(np.unique(coords[:, 0])) * 2)
    
    def record(self, t, **fields):
        """
        Record multiple fields: probe_manager.record(t, pressure=p, chloride=c, ...)
        Special handling for 'water_table' field (computes from pressure)
        """
        if not self.times or self.times[-1] != t:
            self.times.append(t)
        
        for field_name, field_func in fields.items():
            if field_name not in self.fields:
                self.fields[field_name] = {name: [] for name in self.names}
            
            for x_pos, name in zip(self.x_positions, self.names):
                if field_name == 'water_table':
                    val = self._find_water_table(field_func, x_pos)
                else:
                    val = self._sample_at_point(field_func, x_pos)
                self.fields[field_name][name].append(val)
    
    def _sample_at_point(self, field_func, x_pos):
        """Sample field at (x_pos, reference_y)"""
        try:
            return float(field_func.at([x_pos, self.reference_y]))
        except:
            return np.nan
    
    def _find_water_table(self, pressure_field, x_pos):
        """Find y where pressure=0 at given x"""
        coords = self.mesh.coordinates.dat.data
        p_vals = pressure_field.dat.data[:]
        
        mask = np.abs(coords[:, 0] - x_pos) < self.x_tol
        if not np.any(mask):
            return np.nan
        
        y_coords, p_at_x = coords[mask, 1], p_vals[mask]
        sort_idx = np.argsort(y_coords)
        y_sorted, p_sorted = y_coords[sort_idx], p_at_x[sort_idx]
        
        if np.all(p_sorted > 0):
            return float(y_sorted[-1])
        if np.all(p_sorted < 0):
            return float(y_sorted[0])
        
        # Find zero crossing
        for i in range(len(p_sorted) - 1):
            if (p_sorted[i] <= 0 <= p_sorted[i+1]) or (p_sorted[i] >= 0 >= p_sorted[i+1]):
                if abs(p_sorted[i+1] - p_sorted[i]) > 1e-14:
                    return float(y_sorted[i] + (0 - p_sorted[i]) * (y_sorted[i+1] - y_sorted[i]) / 
                                (p_sorted[i+1] - p_sorted[i]))
        return np.nan
    
    def get_data(self, field_name=None):
        """Get data for specific field or all fields"""
        times_array = np.array(self.times)
        if field_name:
            return {'times': times_array, 'data': self.fields.get(field_name, {})}
        return {'times': times_array, 'fields': self.fields}


class SnapshotManager:
    """Store complete field snapshots at specific times"""
    
    def __init__(self, snapshot_times, record_all_interval=None):
        """
        Initialize snapshot manager
        
        Args:
            snapshot_times: List of specific times (s) for main snapshots (for plots)
            record_all_interval: If set, record ALL snapshots every N seconds (for GIFs)
                                Example: 1800 = every 30 minutes
        """
        self.requested_times = sorted(snapshot_times)
        self.snapshots = {}  # Main snapshots for plots: {time: {field_name: Function}}
        self.all_snapshots = {}  # All snapshots for GIF: {time: {field_name: Function}}
        self.record_all_interval = record_all_interval
        self.last_all_record_time = -float('inf')

    def should_record(self, t, dt):
        """Check if should save main snapshot now"""
        if not self.requested_times:
            return False
        closest = min(self.requested_times, key=lambda req_t: abs(t - req_t))
        if abs(t - closest) < dt * 0.6:
            self.requested_times.remove(closest)
            return True
        return False
    
    def should_record_all(self, t):
        """Check if should save to all_snapshots (for GIF)"""
        if self.record_all_interval is None:
            return False
        return (t - self.last_all_record_time) >= self.record_all_interval
    
    def record(self, t, **fields):
        """
        Save main snapshot: snapshot_manager.record(t, saturation=sat, pressure=p, ...)
        This goes into self.snapshots for plotting
        """
        from firedrake import Function
        if t not in self.snapshots:
            self.snapshots[t] = {}
        for field_name, field_func in fields.items():
            # Deep copy the function
            self.snapshots[t][field_name] = Function(field_func.function_space()).assign(field_func)
    
    def record_all(self, t, **fields):
        """
        Save to all_snapshots: snapshot_manager.record_all(t, saturation=sat, ...)
        This goes into self.all_snapshots for GIF animations
        """
        from firedrake import Function
        if t not in self.all_snapshots:
            self.all_snapshots[t] = {}
        for field_name, field_func in fields.items():
            # Deep copy the function
            self.all_snapshots[t][field_name] = Function(field_func.function_space()).assign(field_func)
        self.last_all_record_time = t
