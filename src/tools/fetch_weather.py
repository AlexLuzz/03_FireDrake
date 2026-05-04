import pandas as pd
import requests
from io import StringIO
import matplotlib.pyplot as plt

def fetch_weather_data(start_date, end_date, include_temp=False):
    """Fetches daily snow depth and rain data from Meteostat / weather.gc.ca"""
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    def get_daily_data(station_id, year):
        url = (
            "https://climate.weather.gc.ca/climate_data/bulk_data_e.html?"
            f"format=csv&stationID={station_id}&Year={year}&Month=1&Day=1&timeframe=2"
        )
        r = requests.get(url)
        r.raise_for_status()   
        return pd.read_csv(StringIO(r.text))

    def get_daily_range(station_id, start_year, end_year):
        frames = []
        for y in range(start_year, end_year + 1):
            print(f"Fetching {y}...")
            frames.append(get_daily_data(station_id, y))
        return pd.concat(frames, ignore_index=True)

    station_id = 51157   
    df = get_daily_range(station_id, start_date.year, end_date.year)
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    date_mask = (df['Date/Time'] >= start_date) & (df['Date/Time'] <= end_date)
    filtered_df = df[date_mask]

    snow_data = filtered_df['Total Snow (cm)'].fillna(0)
    rain_data = filtered_df['Total Rain (mm)'].fillna(0)
    temp_data = filtered_df['Mean Temp (°C)'].fillna(0) if include_temp else None

    snow_df = pd.DataFrame({'date': filtered_df['Date/Time'], 'snow': snow_data})
    rain_df = pd.DataFrame({'date': filtered_df['Date/Time'], 'rain': rain_data})
    
    if include_temp:
        temp_df = pd.DataFrame({'date': filtered_df['Date/Time'], 'temp': temp_data})
        return rain_df, snow_df, temp_df
    return rain_df, snow_df

def plot_weather_data(rain_df, snow_df):
    """Visualizes the imported weather data."""
    fig, ax1 = plt.subplots(figsize=(10, 5))
    
    ax1.bar(rain_df['date'], rain_df['rain'], color='blue', label='Rain (mm)', alpha=0.6)
    ax1.set_ylabel('Rain (mm)', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    ax2.plot(snow_df['date'], snow_df['snow'], color='cyan', label='Snow (cm)', linewidth=2)
    ax2.set_ylabel('Snow Depth (cm)', color='cyan')
    ax2.tick_params(axis='y', labelcolor='cyan')
    
    plt.title("Imported Meteo Data")
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test block to visualize data upon execution
    print("Fetching sample data...")
    rain, snow = fetch_weather_data('2024-01-01', '2024-04-01')
    print(f"Imported {len(rain)} days of data.")
    plot_weather_data(rain, snow)