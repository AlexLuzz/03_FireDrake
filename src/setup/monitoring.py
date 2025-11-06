"""
Generic monitoring for time series and spatial snapshots
"""
import numpy as np
from firedrake import Function, FunctionSpace, VertexOnlyMesh, interpolate, Constant, assemble

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
        
        # Default probe positions
        if probe_positions is None:
            probe_positions = [[8.0, 1.0], [10.0, 1.0], [12.5, 1.0]]
        
        self.probe_positions = probe_positions
        self.names = names or [f"Probe_{i+1}" for i in range(len(probe_positions))]
        
        # Create vertex-only mesh at probe locations (stays on tape!)
        self.vom = VertexOnlyMesh(mesh, probe_positions)
        self.P0DG = FunctionSpace(self.vom, "DG", 0)

        # Data storage: {probe_name: {field_name: [values]}}
        self.times = []
        self.data = {name: {} for name in self.names}
        self.data_func = []

        # Global (non-probe) time series storage: {series_name: [values]}
        self.global_data = {}
        
        self.coords = self.mesh.coordinates.dat.data_ro
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

    def record_old(self, t: float, field: Function = None, field_name: str = "value", data: float = None):
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

        # Case 1: Firedrake Function provided -> sample at probe nodes
        if isinstance(field, Function):
            field_data = field.dat.data_ro
            for name, idx in zip(self.names, self._probe_node_indices):
                if field_name not in self.data[name]:
                    self.data[name][field_name] = []
                self.data[name][field_name].append(float(field_data[idx]))
            return
        # Case 2: Generic scalar/array provided (global time series)
        if data is None and field is not None:
            # Support positional use: record(t, value, "series_name")
            data = field
        if data is not None:
            if field_name not in self.global_data:
                self.global_data[field_name] = []
            self.global_data[field_name].append(float(data))
            return
    
    def record(self, t: float, field: Function = None, field_name: str = "value", 
               data: float = None, store_function: bool = True):
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
        data : float
            Generic scalar/array for global time series
        store_function : bool
            If True, also store the interpolated Function (for adjoint)
        """
        # Add time if this is a new timestep
        if not self.times or self.times[-1] != t:
            self.times.append(t)

        # Case 1: Firedrake Function provided -> interpolate to probe points
        if isinstance(field, Function):
            # Use VertexOnlyMesh interpolation (stays on adjoint tape!)
            field_at_probes = assemble(interpolate(field, self.P0DG))

            # Convert pressure head to total head by adding probe elevation (only for water table)
            if field_name == "water_table":
                probe_y_func = Function(self.P0DG)
                probe_y_func.dat.data[:] = [probe[1] for probe in self.probe_positions]
                # Need to project the sum back to a Function for .dat access
                field_with_elevation = Function(self.P0DG)
                field_with_elevation.assign(field_at_probes + probe_y_func)
                field_at_probes = field_with_elevation
            
            probe_values = field_at_probes.dat.data_ro
            
            # Store values in data dict
            for name, val in zip(self.names, probe_values):
                if field_name not in self.data[name]:
                    self.data[name][field_name] = []
                self.data[name][field_name].append(float(val))
            
            # Optionally store Function for adjoint
            if store_function:
                self.data_func.append([t, field_name, field_at_probes])
            return
        
        # Case 2: Generic scalar/array provided (global time series)
        if data is None and field is not None:
            data = field
        if data is not None:
            if field_name not in self.global_data:
                self.global_data[field_name] = []
            self.global_data[field_name].append(float(data))
            return
    
    def get_probe_data(self, probe_name: str, field_name: str):
        """Get data for specific probe and field"""
        return np.array(self.data[probe_name][field_name])
    
    def get_data(self):
        """Get all recorded data"""
        return {'times': np.array(self.times), 'data': self.data, 'global': self.global_data}
    
    def get_recorded_functions(self):
        """Get stored Functions for adjoint optimization"""
        return self.data_func
    
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
            # Include global series
            for field_name in self.global_data.keys():
                header.append(f'global_{field_name}')
            writer.writerow(header)
            
            # Write data
            for i, t in enumerate(times_hours):
                row = [t]
                for name in self.names:
                    for field_name in self.data[name].keys():
                        row.append(self.data[name][field_name][i])
                for field_name in self.global_data.keys():
                    row.append(self.global_data[field_name][i] if i < len(self.global_data[field_name]) else '')
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