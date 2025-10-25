"""
Fetch rainfall data from Meteostat and create source term scenario
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Stations, Hourly

# Import your source term module
# from source_term_improved import rainfall_scenario
# For demonstration, we'll create a mock version


def fetch_and_aggregate_weather(start_date, end_date, temp_step=1, precip_step=24):
    """
    Fetches hourly weather data and aggregates temperature and precipitation.
    
    Parameters:
        start_date (str or datetime): Start date.
        end_date (str or datetime): End date.
        temp_step (int): Step size for temperature aggregation (hours).
        precip_step (int): Step size for precipitation aggregation (hours).
    
    Returns:
        tuple: (temp_df, precip_df) - DataFrames with aggregated data
    """
    # Convert start_date and end_date to datetime objects
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Get Montreal station
    stations = Stations()
    station = stations.region('CA', 'QC')
    
    # Fetch hourly weather data for the specified date range
    data = Hourly('SOK6B', start_date, end_date)
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
    
    return temp_df, precip_df


def create_rainfall_csv(precip_df, output_path='rainfall_data.csv'):
    """
    Convert precipitation DataFrame to CSV format for source_term module.
    
    Args:
        precip_df: DataFrame with 'date' and 'precipitation' columns
        output_path: Path to save CSV file
    
    Returns:
        str: Path to created CSV file
    """
    # Calculate rainfall intensity (mm/hr)
    # Meteostat gives accumulated mm over the aggregation period
    time_diffs = precip_df['date'].diff().dt.total_seconds() / 3600  # hours
    time_diffs.iloc[0] = time_diffs.iloc[1]  # Fill first value
    
    precip_df['intensity_mm_hr'] = precip_df['precipitation'] / time_diffs
    
    # Prepare CSV
    csv_data = precip_df[['date', 'intensity_mm_hr']].copy()
    csv_data.columns = ['Date', 'Pluie tot. (mm)']
    
    # Save to CSV
    csv_data.to_csv(output_path, index=False)
    print(f"✓ Saved rainfall data to {output_path}")
    print(f"  Total precipitation: {precip_df['precipitation'].sum():.1f} mm")
    print(f"  Max intensity: {precip_df['intensity_mm_hr'].max():.1f} mm/hr")
    print(f"  Number of records: {len(precip_df)}")
    
    return output_path


def plot_rainfall_comparison(precip_df, scenario=None, domain_length=100):
    """
    Plot rainfall data and optionally compare with scenario events.
    
    Args:
        precip_df: DataFrame with precipitation data
        scenario: SourceScenario object (optional)
        domain_length: Domain length for scenario creation
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Calculate intensity
    time_diffs = precip_df['date'].diff().dt.total_seconds() / 3600
    time_diffs.iloc[0] = time_diffs.iloc[1]
    intensity = precip_df['precipitation'] / time_diffs
    
    # Plot 1: Raw precipitation (mm per period)
    ax1 = axes[0]
    ax1.bar(precip_df['date'], precip_df['precipitation'], 
            width=time_diffs.iloc[0]/24, alpha=0.7, color='steelblue', 
            label=f'Accumulated (mm per {int(time_diffs.iloc[0])}h)')
    ax1.set_ylabel('Precipitation (mm)', fontsize=12, fontweight='bold')
    ax1.set_title('Raw Meteostat Data', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Intensity (mm/hr) - what goes to solver
    ax2 = axes[1]
    
    # Plot as step function (constant over each period)
    for i in range(len(precip_df)):
        start = precip_df['date'].iloc[i]
        if i < len(precip_df) - 1:
            end = precip_df['date'].iloc[i + 1]
        else:
            end = start + pd.Timedelta(hours=time_diffs.iloc[i])
        
        rate = intensity.iloc[i]
        if rate > 0:
            ax2.plot([start, end], [rate, rate], 'b-', linewidth=2, alpha=0.7)
            ax2.plot([start, start], [0, rate], 'b--', linewidth=1, alpha=0.5)
            ax2.plot([end, end], [rate, 0], 'b--', linewidth=1, alpha=0.5)
    
    # Optionally plot scenario events
    if scenario:
        ax2_twin = ax2.twinx()
        for event in scenario.events:
            if event.start_datetime and event.end_datetime:
                start = event.start_datetime
                end = event.end_datetime
                rate = event.rate
                ax2_twin.plot([start, end], [rate, rate], 'r-', 
                            linewidth=3, alpha=0.6, label='Scenario events')
        ax2_twin.set_ylabel('Scenario Rate (mm/hr)', color='r', fontsize=12, fontweight='bold')
        ax2_twin.tick_params(axis='y', labelcolor='r')
    
    ax2.set_ylabel('Intensity (mm/hr)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax2.set_title('Rainfall Intensity (Constant over periods → Solver Input)', 
                  fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Format x-axis
    fig.autofmt_xdate()
    plt.tight_layout()
    
    return fig, axes


def analyze_aggregation_periods(start_date, end_date, periods=[4, 6, 8, 12, 24]):
    """
    Compare different aggregation periods and their effect on rainfall representation.
    
    Args:
        start_date: Start date for data fetch
        end_date: End date for data fetch
        periods: List of aggregation periods (hours) to compare
    """
    fig, axes = plt.subplots(len(periods), 1, figsize=(14, 3*len(periods)), sharex=True)
    
    for idx, period in enumerate(periods):
        print(f"\n{'='*60}")
        print(f"Aggregation period: {period} hours")
        print(f"{'='*60}")
        
        # Fetch data
        temp_df, precip_df = fetch_and_aggregate_weather(start_date, end_date, 
                                                         precip_step=period)
        
        # Calculate intensity
        time_diffs = precip_df['date'].diff().dt.total_seconds() / 3600
        time_diffs.iloc[0] = time_diffs.iloc[1]
        intensity = precip_df['precipitation'] / time_diffs
        
        # Plot
        ax = axes[idx] if len(periods) > 1 else axes
        
        # Step plot showing constant intensities
        for i in range(len(precip_df)):
            start = precip_df['date'].iloc[i]
            if i < len(precip_df) - 1:
                end = precip_df['date'].iloc[i + 1]
            else:
                end = start + pd.Timedelta(hours=time_diffs.iloc[i])
            
            rate = intensity.iloc[i]
            if rate > 0:
                ax.fill_between([start, end], 0, rate, alpha=0.5, step='post')
                ax.plot([start, end], [rate, rate], 'b-', linewidth=2)
        
        ax.set_ylabel(f'{period}h\nIntensity (mm/hr)', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(bottom=0)
        
        # Stats
        max_intensity = intensity.max()
        total_precip = precip_df['precipitation'].sum()
        rain_events = (intensity > 0).sum()
        
        ax.text(0.02, 0.95, 
               f'Max: {max_intensity:.1f} mm/hr | Total: {total_precip:.1f} mm | Events: {rain_events}',
               transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        print(f"  Total precipitation: {total_precip:.1f} mm")
        print(f"  Max intensity: {max_intensity:.1f} mm/hr")
        print(f"  Number of rain events: {rain_events}")
    
    axes[0].set_title('Rainfall Intensity at Different Aggregation Periods', 
                     fontsize=14, fontweight='bold')
    axes[-1].set_xlabel('Date', fontsize=12, fontweight='bold')
    fig.autofmt_xdate()
    plt.tight_layout()
    
    return fig, axes


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("METEOSTAT RAINFALL DATA → SOURCE TERM SCENARIO")
    print("="*70)
    
    # Configuration
    START_DATE = '2024-05-01'
    END_DATE = '2024-05-15'
    AGGREGATION_PERIOD = 6  # hours
    DOMAIN_LENGTH = 100  # meters
    
    print(f"\nFetching data from {START_DATE} to {END_DATE}")
    print(f"Aggregation period: {AGGREGATION_PERIOD} hours")
    
    # Fetch weather data
    try:
        temp_df, precip_df = fetch_and_aggregate_weather(
            START_DATE, END_DATE, 
            precip_step=AGGREGATION_PERIOD
        )
        
        print(f"\n✓ Fetched {len(precip_df)} precipitation records")
        
        # Create CSV for source_term module
        csv_path = create_rainfall_csv(precip_df, 'rainfall_meteostat.csv')
        
        # Plot the data
        print("\nGenerating plots...")
        fig1, _ = plot_rainfall_comparison(precip_df, domain_length=DOMAIN_LENGTH)
        plt.savefig('rainfall_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Saved plot to rainfall_analysis.png")
        
        # Compare different aggregation periods
        print("\n" + "="*70)
        print("COMPARING AGGREGATION PERIODS")
        print("="*70)
        
        fig2, _ = analyze_aggregation_periods(START_DATE, END_DATE, 
                                              periods=[4, 6, 8, 12])
        plt.savefig('rainfall_aggregation_comparison.png', dpi=150, bbox_inches='tight')
        print("\n✓ Saved comparison to rainfall_aggregation_comparison.png")
        
        # Show how to use with source_term module
        print("\n" + "="*70)
        print("TO USE WITH SOURCE TERM MODULE:")
        print("="*70)
        print("""
from source_term_improved import rainfall_scenario
from datetime import datetime

# Create scenario from CSV
scenario = rainfall_scenario(
    domain_length=100,  # Domain size in meters
    csv_path='rainfall_meteostat.csv',
    time_col='Date',
    rain_col='Pluie tot. (mm)',
    from_date=datetime(2024, 5, 1),
    to_date=datetime(2024, 5, 15),
    rain_unit='mm/hour',  # Already converted to mm/hr
    time_unit='days'
)

# Use in FEM solver
flux_expr = scenario.get_flux_expression(mesh, t_seconds)
        """)
        
        plt.show()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure you have meteostat installed:")
        print("  pip install meteostat")
        import traceback
        traceback.print_exc()