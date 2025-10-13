# monitoring/probes.py
"""
Monitoring utilities for time series and spatial snapshots
"""
import numpy as np

class ProbeManager:
    """Manages water table elevation monitoring at specific x-locations"""
    
    def __init__(self, mesh, x_positions, names=None):
        self.mesh = mesh
        self.x_positions = x_positions
        self.names = names or [f"Probe_{i+1}" for i in range(len(x_positions))]
        self.data = {name: [] for name in self.names}
        self.times = []
        
        # Mesh tolerance
        coords = mesh.coordinates.dat.data
        self.x_tol = (coords[:, 0].max() - coords[:, 0].min()) / (len(np.unique(coords[:, 0])) * 2)
    
    def find_water_table_at_x(self, pressure_field, x_position):
        """Find water table elevation (where p=0) at given x position"""
        coords = self.mesh.coordinates.dat.data
        p_vals = pressure_field.dat.data[:]
        
        mask = np.abs(coords[:, 0] - x_position) < self.x_tol
        if not np.any(mask):
            return None
        
        y_coords = coords[mask, 1]
        p_at_x = p_vals[mask]
        
        sort_idx = np.argsort(y_coords)
        y_sorted = y_coords[sort_idx]
        p_sorted = p_at_x[sort_idx]
        
        if np.all(p_sorted > 0):
            return y_sorted[-1]
        if np.all(p_sorted < 0):
            return y_sorted[0]
        
        for i in range(len(p_sorted) - 1):
            if p_sorted[i] <= 0 and p_sorted[i+1] >= 0:
                y1, y2 = y_sorted[i], y_sorted[i+1]
                p1, p2 = p_sorted[i], p_sorted[i+1]
                if p2 - p1 != 0:
                    return y1 - p1 * (y2 - y1) / (p2 - p1)
                return y1
        return None
    
    def record(self, t, pressure_field):
        """Record water table elevation at all probe locations"""
        self.times.append(t)
        for x_pos, name in zip(self.x_positions, self.names):
            wt = self.find_water_table_at_x(pressure_field, x_pos)
            self.data[name].append(wt if wt is not None else np.nan)
    
    def record_initial(self, pressure_field):
        """Record initial state at t=0 (call before time loop)"""
        self.record(0.0, pressure_field)
    
    def get_data(self):
        """Get monitoring data"""
        return {'times': np.array(self.times), 'data': self.data}
    
    def get_time_hours(self):
        """Get times in hours"""
        return np.array(self.times) / 3600.0
    
    def get_water_table_depth(self, name, reference_elevation=None):
        """Get depth to water table (positive = below reference)"""
        if reference_elevation is None:
            coords = self.mesh.coordinates.dat.data
            reference_elevation = coords[:, 1].max()
        return reference_elevation - np.array(self.data[name])
    
    def compare_with_measurements(self, measured_times, measured_depths, 
                                  probe_name, reference_elevation=None):
        """Compare simulated with measured data"""
        times_hours = self.get_time_hours()
        sim_depths = self.get_water_table_depth(probe_name, reference_elevation)
        sim_at_measured = np.interp(measured_times, times_hours, sim_depths)
        
        rmse = np.sqrt(np.mean((sim_at_measured - measured_depths)**2))
        mae = np.mean(np.abs(sim_at_measured - measured_depths))
        bias = np.mean(sim_at_measured - measured_depths)
        correlation = np.corrcoef(sim_at_measured, measured_depths)[0, 1]
        
        return {
            'rmse': rmse, 'mae': mae, 'bias': bias, 'correlation': correlation,
            'simulated_at_measured_times': sim_at_measured
        }
    
    def save_to_csv(self, filename, reference_elevation=None):
        """Save water table data to CSV"""
        import csv
        times_hours = self.get_time_hours()
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            if reference_elevation is None:
                header = ['time_hours'] + [f'{name}_elevation_m' for name in self.names]
            else:
                header = ['time_hours'] + \
                        [f'{name}_elevation_m' for name in self.names] + \
                        [f'{name}_depth_m' for name in self.names]
            writer.writerow(header)
            
            for i, t in enumerate(times_hours):
                row = [t] + [self.data[name][i] for name in self.names]
                if reference_elevation is not None:
                    row += [reference_elevation - self.data[name][i] for name in self.names]
                writer.writerow(row)
        
        print(f"Water table data saved to {filename}")


class SnapshotManager:
    """Manages spatial snapshots at specific times"""
    
    def __init__(self, snapshot_times, domain):
        self.snapshot_times = snapshot_times
        self.snapshots = {}
        self.domain = domain
        self.initial_recorded = False
    
    def should_record(self, t, dt):
        """Check if current time is close to a snapshot time"""
        return any(abs(t - st) < dt/2 for st in self.snapshot_times)
    
    def record(self, t, pressure_field):
        """Record snapshot at current time"""
        saturation = self.domain.compute_saturation_field(pressure_field)
        
        self.snapshots[t] = {
            'pressure': pressure_field.copy(deepcopy=True),
            'saturation': saturation.copy(deepcopy=True)
        }
        
        sat_vals = saturation.dat.data[:]
        p_vals = pressure_field.dat.data[:]
        print(f"  Saved snapshot at t={t/3600:.2f}h")
        print(f"    Saturation: [{sat_vals.min():.4f}, {sat_vals.max():.4f}]")
        print(f"    Pressure: [{p_vals.min():.4f}, {p_vals.max():.4f}]")
    
    def record_initial(self, pressure_field):
        """Record initial state at t=0 (call before time loop)"""
        if not self.initial_recorded:
            self.record(0.0, pressure_field)
            self.initial_recorded = True
    
    def get_sorted_times(self):
        """Get sorted list of snapshot times"""
        return sorted(self.snapshots.keys())