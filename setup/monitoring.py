# monitoring/probes.py
"""
Monitoring utilities for time series and spatial snapshots
"""
import numpy as np

class ProbeManager:
    """Manages water table elevation monitoring at specific x-locations"""
    
    def __init__(self, mesh, probes_positions, names=None):
        self.mesh = mesh
        self.probes_positions = probes_positions
        self.names = names or [f"Probe_{i+1}" for i in range(len(probes_positions))]
        self.data = {name: [] for name in self.names}
        self.times = []
        
        # Fixed: Use 1% of domain width as tolerance (more robust)
        coords = mesh.coordinates.dat.data
        domain_width = coords[:, 0].max() - coords[:, 0].min()
        self.x_tol = domain_width * 0.01  # 1% of domain width
        
        print(f"ProbeManager initialized with x_tol = {self.x_tol:.4f}m")

    def find_water_table_at_probe_pos(self, pressure_field, probe_pos):
        """Find water table elevation (where p=0) at given probe position"""
        coords = self.mesh.coordinates.dat.data
        p_vals = pressure_field.dat.data[:]

        x_pos, y_pos = probe_pos
        
        # Find all nodes near this x position
        mask = np.abs(coords[:, 0] - x_pos) < self.x_tol
        if not np.any(mask):
            print(f"⚠️  No nodes found near x={x_pos:.2f}m (tol={self.x_tol:.4f}m)")
            return None
        
        # Get y-coordinates and pressures at this x
        y_coords = coords[mask, 1]  # Column 1 = y coordinates
        p_at_x = p_vals[mask]
        
        # Sort by y (bottom to top)
        sort_idx = np.argsort(y_coords)
        y_sorted = y_coords[sort_idx]
        p_sorted = p_at_x[sort_idx]
        
        # Case 1: Fully saturated column
        if np.all(p_sorted > 0):
            return float(y_sorted[-1])  # Water table at surface
        
        # Case 2: Fully unsaturated column
        if np.all(p_sorted < 0):
            return float(y_sorted[0])  # Water table at bottom
        
        # Case 3: Find zero crossing (linear interpolation)
        for i in range(len(p_sorted) - 1):
            p1, p2 = p_sorted[i], p_sorted[i+1]
            y1, y2 = y_sorted[i], y_sorted[i+1]
            
            # Check if zero is between these two points
            if (p1 <= 0 <= p2) or (p2 <= 0 <= p1):
                if abs(p2 - p1) > 1e-12:
                    # Linear interpolation: y_wt = y1 + (0 - p1) * (y2 - y1) / (p2 - p1)
                    wt_y = y1 - p1 * (y2 - y1) / (p2 - p1)
                    return float(wt_y)
                else:
                    return float((y1 + y2) / 2.0)
        
        # Fallback: shouldn't reach here
        return float(y_sorted[len(y_sorted)//2])
    
    def record(self, t, pressure_field):
        """Record water table elevation at all probe locations"""
        self.times.append(t)
        for probe_pos, name in zip(self.probes_positions, self.names):
            wt = self.find_water_table_at_probe_pos(pressure_field, probe_pos)
            self.data[name].append(wt if wt is not None else np.nan)
    
    def record_initial(self, pressure_field):
        """Record initial state at t=0"""
        self.record(0.0, pressure_field)
    
    def get_data(self):
        """Get monitoring data"""
        return {'times': np.array(self.times), 'data': self.data}
    
    def save_to_csv(self, filename):
        """Save water table data to CSV"""
        import csv
        times_hours = np.array(self.times) / 3600.0
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            header = ['time_hours'] + [f'{name}_elevation_m' for name in self.names]
            writer.writerow(header)
            
            for i, t in enumerate(times_hours):
                row = [t] + [self.data[name][i] for name in self.names]
                writer.writerow(row)
        
        print(f"Water table data saved to {filename}")


class SnapshotManager:
    """Manages spatial snapshots at specific times"""
    
    def __init__(self, snapshot_times, domain):
        self.snapshot_times = list(snapshot_times)
        self.requested_times = list(snapshot_times)
        self.snapshots = {}
        self.domain = domain
    
    def should_record(self, t, dt):
        """Check if current time is close to a snapshot time"""
        for req_time in self.requested_times[:]:
            if abs(t - req_time) < dt * 0.6:
                self.requested_times.remove(req_time)
                return True
        return False
    
    def record(self, t, pressure_field):
        """Record snapshot at current time"""
        saturation = self.domain.compute_saturation_field(pressure_field)
        
        self.snapshots[t] = {
            'pressure': pressure_field.copy(deepcopy=True),
            'saturation': saturation.copy(deepcopy=True)
        }
        
        sat_vals = saturation.dat.data[:]
        print(f"  Snapshot at t={t/3600:.2f}h: S=[{sat_vals.min():.3f}, {sat_vals.max():.3f}]")
    
    def record_initial(self, pressure_field):
        """Record initial state at t=0"""
        if 0.0 not in self.snapshots:
            self.record(0.0, pressure_field)