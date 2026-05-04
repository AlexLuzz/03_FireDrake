from firedrake import Constant, SpatialCoordinate, conditional
import pandas as pd
import numpy as np
from ..tools.csv_loader import CSVLoader 
from ..tools.fetch_weather import fetch_weather_data

class SourceManager:
    def __init__(self, config, zones: dict):
        """
        config: SimulationConfig instance
        zones: {'zone_name': (x_min, x_max, y_min, y_max)}
        """
        self.config = config
        self.zones = zones
        # Dataframe stores temporal source fluxes at simulation time scale
        self.data = pd.DataFrame(0.0, index=self.config.time_steps, columns=[])

    def add_event(self, column: str, start_h, end_h, value: float):
        """Manual event definition using hours from simulation start."""
        if column not in self.data.columns:
            self.data[column] = 0.0
            
        mask = (self.data.index >= start_h * 3600) & (self.data.index < end_h * 3600)
        self.data.loc[mask, column] = value

    def import_csv(self, filepath: str, datetime_col: str = None):
        """
        Imports raw time-series data. 
        IMPORTANT: All numeric values in the CSV columns must be in SI units (m/s).
        """
        loader = CSVLoader(filepath, datetime_col)
        df = loader.df
        
        # Align time to simulation seconds
        if isinstance(df.index, pd.DatetimeIndex):
            sim_t = (df.index - self.config.start_datetime).total_seconds()
        else:
            sim_t = df.index * 3600.0 # Assumes numeric index is hours
            
        df.index = sim_t
        # Reindex and forward-fill to align with simulation time steps
        new_data = df.reindex(self.config.time_steps).ffill().fillna(0.0)
        self.data = pd.concat([self.data, new_data], axis=1)

    def import_weather_data(self, start_date, end_date):
        """Fetches station data and converts units to SI (m/s)."""
        rain_df, snow_df = fetch_weather_data(start_date, end_date)
        
        for df, col, factor in [(rain_df, 'rain', 1/(1000*86400)), 
                                (snow_df, 'snow', 1/(100*86400))]:
            df['sim_t'] = (df['date'] - self.config.start_datetime).dt.total_seconds()
            series = df.set_index('sim_t')[col]
            self.data[col] = series.reindex(self.config.time_steps).ffill().fillna(0.0).values * factor

    def get_flux_expression(self, t: float, mesh):
        """
        Constructs the UFL expression by applying zone weighting to fluxes.
        """
        x, y = SpatialCoordinate(mesh)
        expr = Constant(0.0)
        
        # Get the current flux values for all columns at time t[cite: 4]
        idx = (np.abs(self.config.time_steps - t)).argmin()
        current_step = self.data.iloc[idx]
        
        for name, info in self.zones.items():
            col = info['data_col']
            if col in current_step:
                # Raw value from dataframe * zone-specific factor
                val = current_step[col] * info.get('factor', 1.0)
                
                if val == 0.0: continue
                
                # Spatial bounds logic
                b = info['bounds']
                cond = (x >= b[0]) & (x <= b[1])
                if len(b) > 2 and b[2] is not None:
                    cond &= (y >= b[2]) & (y <= b[3])
                
                expr = conditional(cond, Constant(val), expr)
        return expr