import sys
import os
# Add the parent directory to Python path to import the main package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *

config = SimulationConfig(
        project_name="Test",
        user="alexi",
        start_datetime=datetime(2024, 4, 15),
        end_datetime=datetime(2024, 5, 30),
        dt_td=timedelta(hours=3)
    )
    
rain_source = rainfall_scenario(
        from_date=config.start_datetime,
        to_date=config.end_datetime,
        # From CSV file (need to specify path and rain unit)
        csv_path=config.data_input_dir / "BB_METEO.csv",
        rain_unit="mm/day",
        # From Meteostat (uncomment to use)
        #meteostat_station='SOK6B',
        #meteostat_agg_hours=6,
    )