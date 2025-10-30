"""
Generic monitoring for time series and spatial snapshots
"""
import numpy as np

class ProbeManager:
    """
    Monitor field values at specific spatial locations over time
    Generic: works for pressure, concentration, temperature, etc.
    """
    
    def __init__(self, mesh, probe_positions=None, names=None):
        """
        Parameters:
        -----------
        mesh : Firedrake mesh
        probe_positions : list of [x, y] coordinates
        names : list of probe names
        """
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data
        
        # Default probe positions
        if probe_positions is None:
            probe_positions = [[8.0, 1.0], [10.0, 1.0], [12.5, 1.0]]
        
        self.probe_positions = probe_positions
        self.names = names or [f"Probe_{i+1}" for i in range(len(probe_positions))]
        
        # Data storage: {probe_name: {field_name: [values]}}
        self.data = {name: {} for name in self.names}
        self.times = []
        
        # Tolerance for finding nodes (1% of domain width)
        domain_width = self.coords[:, 0].max() - self.coords[:, 0].min()
        self.x_tol = domain_width * 0.01
        
        # Cache node indices for efficiency
        self._probe_node_indices = self._find_probe_nodes()
    
    def _find_probe_nodes(self):
        """Find nearest mesh node for each probe (cached for efficiency)"""
        indices = []
        for probe_pos in self.probe_positions:
            x_probe, y_probe = probe_pos
            # Find closest node
            distances = np.sqrt((self.coords[:, 0] - x_probe)**2 + 
                              (self.coords[:, 1] - y_probe)**2)
            idx = np.argmin(distances)
            indices.append(idx)
        return indices
    
    def record(self, t: float, field, field_name: str = "value"):
        """
        Record field values at all probe locations (GENERIC)
        
        Parameters:
        -----------
        t : float
            Time [s]
        field : Firedrake Function
            Any field to monitor (pressure, concentration, etc.)
        field_name : str
            Name for this field (e.g., "pressure", "concentration")
        """
        # Add time if this is a new timestep
        if not self.times or self.times[-1] != t:
            self.times.append(t)
        
        # Extract values at probe locations
        field_data = field.dat.data_ro
        
        for name, idx in zip(self.names, self._probe_node_indices):
            if field_name not in self.data[name]:
                self.data[name][field_name] = []
            self.data[name][field_name].append(float(field_data[idx]))
    
    def record_water_table(self, t: float, pressure_field):
        """
        Record water table elevation (specialized method)
        Finds where pressure = 0 along vertical at each probe x-position
        """
        if not self.times or self.times[-1] != t:
            self.times.append(t)
        
        p_vals = pressure_field.dat.data_ro
        
        for probe_pos, name in zip(self.probe_positions, self.names):
            wt = self._find_water_table_at_x(probe_pos[0], p_vals)
            
            if "water_table" not in self.data[name]:
                self.data[name]["water_table"] = []
            self.data[name]["water_table"].append(wt)
    
    def _find_water_table_at_x(self, x_pos: float, p_vals: np.ndarray):
        """Find water table elevation at given x-position"""
        # Find nodes near this x
        mask = np.abs(self.coords[:, 0] - x_pos) < self.x_tol
        if not np.any(mask):
            return np.nan
        
        # Get y-coordinates and pressures
        y_coords = self.coords[mask, 1]
        p_at_x = p_vals[mask]
        
        # Sort by y
        sort_idx = np.argsort(y_coords)
        y_sorted = y_coords[sort_idx]
        p_sorted = p_at_x[sort_idx]
        
        # Find zero crossing
        if np.all(p_sorted > 0):
            return float(y_sorted[-1])  # Fully saturated
        if np.all(p_sorted < 0):
            return float(y_sorted[0])   # Fully dry
        
        # Interpolate to find p=0
        for i in range(len(p_sorted) - 1):
            if (p_sorted[i] <= 0 <= p_sorted[i+1]) or (p_sorted[i+1] <= 0 <= p_sorted[i]):
                if abs(p_sorted[i+1] - p_sorted[i]) > 1e-12:
                    wt_y = y_sorted[i] - p_sorted[i] * (y_sorted[i+1] - y_sorted[i]) / (p_sorted[i+1] - p_sorted[i])
                    return float(wt_y)
        
        return float(y_sorted[len(y_sorted)//2])
    
    def get_times_hours(self):
        """Get times in hours"""
        return np.array(self.times) / 3600.0
    
    def get_probe_data(self, probe_name: str, field_name: str):
        """Get data for specific probe and field"""
        return np.array(self.data[probe_name][field_name])
    
    def get_data(self):
        """Get all recorded data"""
        return {'times': np.array(self.times), 'data': self.data}
    
    def save_to_csv(self, filename: str):
        """Save all data to CSV"""
        import csv
        times_hours = self.get_times_hours()
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Build header
            header = ['time_hours']
            for name in self.names:
                for field_name in self.data[name].keys():
                    header.append(f'{name}_{field_name}')
            writer.writerow(header)
            
            # Write data
            for i, t in enumerate(times_hours):
                row = [t]
                for name in self.names:
                    for field_name in self.data[name].keys():
                        row.append(self.data[name][field_name][i])
                writer.writerow(row)
        
        print(f"✓ Probe data saved to {filename}")


class SnapshotManager:
    """
    Record full spatial fields at specific times
    Generic: stores any Firedrake Function
    """
    
    def __init__(self, snapshot_times):
        """
        Parameters:
        -----------
        snapshot_times : list of floats
            Times [s] when snapshots should be recorded
        """
        self.snapshot_times = list(snapshot_times)
        self.snapshots = {}  # {time: {field_name: Function}}
    
    def should_record(self, t: float, dt: float):
        """Check if current time matches a snapshot time"""
        for req_time in self.snapshot_times[:]:
            if abs(t - req_time) < dt * 0.6:
                self.snapshot_times.remove(req_time)
                return True
        return False
    
    def record(self, t: float, field, field_name: str = "field", verbose: bool = True):
        """
        Record a field snapshot (GENERIC)
        
        Parameters:
        -----------
        t : float
            Time [s]
        field : Firedrake Function
            Any field to snapshot
        field_name : str
            Name for this field
        """
        if t not in self.snapshots:
            self.snapshots[t] = {}
        
        self.snapshots[t][field_name] = field.copy(deepcopy=True)
        
        # Print stats
        vals = field.dat.data_ro
        if verbose:
            print(f"  Snapshot at t={t/3600:.2f}h | {field_name}: [{vals.min():.3f}, {vals.max():.3f}]")

    def get_snapshot(self, t: float, field_name: str):
        """Get a specific field at a specific time"""
        return self.snapshots.get(t, {}).get(field_name)
    
    def get_times(self):
        """Get all snapshot times"""
        return sorted(self.snapshots.keys())