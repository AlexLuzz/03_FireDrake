import sys
import os
# Add the parent directory to Python path to import the main package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import *
from datetime import datetime, timedelta
from matplotlib import colors, pyplot as plt

config = SimulationConfig(
        project_name="Test",
        user="AQ96560", # alexi or AQ96560
        start_datetime=datetime(2024, 4, 15),
        end_datetime=datetime(2024, 5, 30),
        dt_td=timedelta(hours=3)
    )
    
station_list = {'McTavish': '71612', 
                'YUL Airport': 'SOK6B', 
                'St-Hubert': '71371', 
                'Riv. des prairies': 'JVOIG'
                }
colors = ['steelblue', 'darkorange', 'green', 'red']

# Plot all of them on the same figure
fig, ax = plt.subplots()
for station_name, station_code in station_list.items():
    rain_source = rainfall_scenario(
        from_date=config.start_datetime,
        to_date=config.end_datetime,
        # From CSV file (need to specify path and rain unit)
        #csv_path=config.data_input_dir / "BB_METEO.csv",
        #rain_unit="mm/day",
        # From Meteostat (uncomment to use)
        meteostat_station=station_code,
        meteostat_agg_hours=int(config.dt_hours),
    )
    
    ax = BasicPlotting.add_rain_bars(ax, rain_source.events, use_datetime=True, 
                                     color=colors.pop(0), label=station_name, twinx=False)

ax.set_title("Rainfall Scenario Comparison", fontsize=14, fontweight='bold')
ax.set_xlabel("Date", fontsize=12, fontweight='bold')
ax.set_ylabel("Rain (mm/hr)", fontsize=12, fontweight='bold')
ax.legend().set_visible(True)
plt.savefig(fname=config.output_dir / "rain_scenario_test.png")