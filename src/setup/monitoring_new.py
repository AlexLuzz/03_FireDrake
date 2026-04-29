import numpy as np
import csv

class SimulationMonitor:
    """
    Unified monitor for time-series point probes and full spatial snapshots.
    
    Data Structures:
    ----------------
    probes:   self.probe_data["Probe_1"]["water_table"] = [val_t1, val_t2, ...]
    global:   self.global_data["mass_balance"] = [val_t1, val_t2, ...]
    snaps:    self.snapshots[time]["pressure"] = FiredrakeFunction
    """
    
    def __init__(self, mesh, probe_positions=None, names=None, snapshot_times=None):
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data_ro
        
        # --- Time-Series Probes Setup ---
        self.probe_positions = probe_positions or [[8.0, 1.0], [10.0, 1.0], [12.5, 1.0]]
        self.names = names or [f"Probe_{i+1}" for i in range(len(self.probe_positions))]
        
        # Cache the nearest mesh node index for each probe (blazing fast lookups later)
        self._probe_indices = self._find_probe_nodes()
        
        self.times = []
        self.probe_data = {name: {} for name in self.names}
        self.global_data = {}
        
        # --- Spatial Snapshots Setup ---
        self.snapshot_times = sorted(list(snapshot_times)) if snapshot_times else []
        self.snapshots = {}

    def _find_probe_nodes(self):
        """Finds the single closest mesh node index for each probe coordinate."""
        indices = []
        for x_p, y_p in self.probe_positions:
            dist_sq = (self.coords[:, 0] - x_p)**2 + (self.coords[:, 1] - y_p)**2
            indices.append(np.argmin(dist_sq))
        return indices

    def record_probe(self, t: float, field=None, field_name: str = "value", data=None):
        """Records a scalar value at current timestep."""
        if not self.times or self.times[-1] != t:
            self.times.append(t)

        # Case 1: Generic global scalar (e.g., rainfall, error norms)
        if data is not None or not hasattr(field, 'dat'):
            val = data if data is not None else field
            self.global_data.setdefault(field_name, []).append(float(val))
            return

        # Case 2: Firedrake Function -> pull directly from cached node index
        field_data = field.dat.data_ro
        for name, idx, pos in zip(self.names, self._probe_indices, self.probe_positions):
            val = float(field_data[idx])
            
            # Convert pressure head to total head using the probe's Y coordinate
            if field_name == "water_table":
                val += pos[1] 
                
            self.probe_data[name].setdefault(field_name, []).append(val)

    def check_and_record_snapshot(self, t: float, dt: float, field, field_name: str = "field"):
        """Checks if current time is a snapshot target, and saves the Function if so."""
        for req_time in self.snapshot_times[:]:
            if abs(t - req_time) < dt * 0.6:
                self.snapshot_times.remove(req_time)
                
                # Deepcopy is required so the Function doesn't overwrite itself next timestep
                self.snapshots.setdefault(t, {})[field_name] = field.copy(deepcopy=True)
                
                vals = field.dat.data_ro
                print(f"  Snapshot at t={t/3600:.2f}h | {field_name}: [{vals.min():.3f}, {vals.max():.3f}]")
                return True
        return False

    def save_probes_to_csv(self, filename: str):
        """Dumps all time-series (probes and global) to a CSV."""
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Dynamic header generation
            header = ['time_s', 'time_hours']
            for name in self.names:
                header.extend([f'{name}_{f}' for f in self.probe_data[name].keys()])
            header.extend([f'global_{f}' for f in self.global_data.keys()])
            writer.writerow(header)
            
            # Write rows
            for i, t in enumerate(self.times):
                row = [t, t / 3600.0]
                for name in self.names:
                    row.extend([self.probe_data[name][f][i] for f in self.probe_data[name].keys()])
                for f in self.global_data.keys():
                    row.append(self.global_data[f][i] if i < len(self.global_data[f]) else '')
                writer.writerow(row)
                
        print(f"✓ Probe data saved to {filename}")