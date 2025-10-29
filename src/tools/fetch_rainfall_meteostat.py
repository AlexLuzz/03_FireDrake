"""
Fetch rainfall data from Meteostat and create source term scenario
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from meteostat import Stations, Hourly
#from setup.source_term import rainfall_scenario

def fetch_and_aggregate_weather(start_date, end_date, station='SOK6B', temp_step=1, precip_step=24):
    """
    Fetches hourly weather data and aggregates temperature and precipitation.
    
    Parameters:
        start_date (str or datetime): Start date.
        end_date (str or datetime): End date.
        station (str): Station ID (default: 'SOK6B' for Montreal YUL)
        temp_step (int): Step size for temperature aggregation (hours).
        precip_step (int): Step size for precipitation aggregation (hours).
    
    Returns:
        tuple: (precip_df, temp_df) - DataFrames with aggregated data
    """
    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Get Montreal station
    stations = Stations()
    station_region = stations.region('CA', 'QC')
    
    # Fetch hourly weather data for the specified date range
    # Available stations:
    # For Mirabel International Airport : 3MXT3
    # FOR YUL : SOK6B
    # For McTavish : 71612
    # For Rivière des Prairies : JVOIG
    # For Saint-Hubert : 71371
    data = Hourly(station, start_date, end_date)
    data = data.fetch()
    
    # Extract precipitation and temperature data
    precipitation = data['prcp'].fillna(0)
    temperature = data['temp']
    times = data.index
    
    # Aggregate data based on the specified steps
    t_agg_temp = times[::temp_step]
    temp_agg = [temperature.iloc[i] for i in range(0, len(temperature), temp_step)]
    
    t_agg_precip = times[::precip_step]
    precip_agg = [sum(precipitation[i:i + precip_step]) for i in range(0, len(precipitation), precip_step)]
    
    # Create DataFrames
    temp_df = pd.DataFrame({'date': t_agg_temp, 'temperature': temp_agg})
    precip_df = pd.DataFrame({'date': t_agg_precip, 'precipitation': precip_agg})
    
    return precip_df, temp_df

def plot_scenario(scenario, title='Rainfall_scenario_meteostat', save=True):
    """Plot a single rainfall scenario"""
    fig, ax = plt.subplots(figsize=(12, 5))
    
    max_rate = 0
    for event in scenario.events:
        if event.rate > 0 and event.start_datetime:
            ax.plot([event.start_datetime, event.end_datetime], 
                   [event.rate, event.rate], linewidth=2)
            ax.fill_between([event.start_datetime, event.end_datetime], 
                          0, event.rate, alpha=0.3)
            max_rate = max(max_rate, event.rate)
    
    # Set reasonable y-axis limits to prevent tick overflow
    if max_rate > 0:
        ax.set_ylim(0, max_rate * 1.1)
        if max_rate > 100:
            ax.locator_params(axis='y', nbins=6)
    
    ax.set_ylabel('Intensity (mm/hr)')
    ax.set_xlabel('Date')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    if save:
        plt.savefig(f"{title}.png")
    return fig, ax

def plot_rainfall_intensity(df, title='Rainfall Intensity', save=True, precip_step=24):
    """Plot rainfall intensity from DataFrame"""
    fig, ax = plt.subplots(figsize=(12, 5))

    # Clean data and check for extreme values
    if len(df) > 0 and 'precipitation' in df.columns:
        precip_data = df['precipitation'].copy()
        
        # Remove any invalid values
        precip_data = precip_data[np.isfinite(precip_data) & (precip_data >= 0)]
        
        if len(precip_data) > 0:
            max_precip = precip_data.max()
            
            # Use cleaned data for plotting
            ax.bar(df['date'], df['precipitation'], color='blue', width=0.2)
            
            # Set reasonable y-axis limits and ticks
            if max_precip > 100:
                ax.locator_params(axis='y', nbins=6)
            ax.set_ylim(0, max_precip * 1.1)
        else:
            ax.text(0.5, 0.5, 'No valid precipitation data', transform=ax.transAxes,
                   ha='center', va='center', fontsize=12)
    
    ax.set_ylabel(f'Intensity (mm/{precip_step}hr)')
    ax.set_xlabel('Date')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    if save:
        plt.savefig(f"{title}.png")
    return fig, ax

def compare_scenarios(scenarios, labels=None, title='Scenario Comparison'):
    """Plot multiple scenarios for comparison"""
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = plt.cm.tab10(range(len(scenarios)))
    labels = labels or [f'Scenario {i}' for i in range(len(scenarios))]
    
    for scenario, label, color in zip(scenarios, labels, colors):
        for event in scenario.events:
            if event.rate > 0 and event.start_datetime:
                ax.plot([event.start_datetime, event.end_datetime], 
                       [event.rate, event.rate], linewidth=2, 
                       color=color, label=label, alpha=0.8)
                label = None  # Only label once
    
    ax.set_ylabel('Intensity (mm/hr)')
    ax.set_xlabel('Date')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Example usage
    START = '2024-04-01'
    END = '2024-07-30'
    precip_step = 6  # 6-hourly precipitation aggregation

    # Fetch data
    df, _ = fetch_and_aggregate_weather(START, END, station='SOK6B', precip_step=precip_step)
    print(f"Fetched {len(df)} records")
    print(f"Max: {df['precipitation'].max():.1f} mm/{precip_step}hr")

    plot_rainfall_intensity(df, title='Rainfall Intensity Meteostat', precip_step=precip_step)
    #plot_scenario(scenario, title='Rainfall Scenario')

    # Save CSV
    csv_data = df[['date', 'precipitation']].copy()
    csv_data.columns = ['Date', 'Pluie tot. (mm)']
    csv_data.to_csv('rainfall.csv', index=False)
    print("Saved to rainfall.csv")