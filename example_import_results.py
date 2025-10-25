"""
Example usage of ImportResults class for data import and visualization
"""
from datetime import datetime
from tools.import_results import ImportResults
from tools.time_converter import TimeConverter

def example_data_import():
    """Example showing how to use ImportResults independently"""
    
    # Create time converter (simulation starts April 1, 2024)
    start_datetime = datetime(2024, 4, 1)
    time_converter = TimeConverter(start_datetime)
    
    # Initialize ImportResults
    importer = ImportResults(time_converter)
    
    # Set up individual piezometer offsets
    # (in real scenarios, piezometers might be at different elevations)
    importer.add_piezometer_offset("LTC 101", vertical_offset=0.60, display_name="LTC 1")
    importer.add_piezometer_offset("LTC 102", vertical_offset=0.65, display_name="LTC 2") 
    importer.add_piezometer_offset("LTC 103", vertical_offset=0.55, display_name="LTC 3")
    
    # Load COMSOL data
    comsol_ref_date = datetime(2024, 2, 22)  # COMSOL t=0 reference
    start_from_days = (start_datetime - comsol_ref_date).days  # ~39 days
    
    comsol_data = importer.load_comsol_data(
        csv_path="data_input/RAF_COMSOL_PZ_CG.csv",
        start_from_days=start_from_days,
        sim_duration_days=29.0,  # April 1-30 = 29 days
        ref_date=comsol_ref_date
    )
    
    # Load measured data with filtering
    end_datetime = datetime(2024, 4, 30)
    measured_data = importer.load_measured_data(
        csv_path="data_input/MEASURED_PZ_CG.csv",
        start_datetime=start_datetime,
        end_datetime=end_datetime,
        apply_offsets=True  # Apply individual offsets
    )
    
    # Get data summary
    summary = importer.get_data_summary()
    print("\\n📊 Data Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Create preview visualization
    if comsol_data or measured_data:
        print("\\n📈 Creating data preview...")
        fig = importer.preview_data(figsize=(14, 8), show_offsets=True)
        fig.savefig("results/data_preview.png", dpi=150, bbox_inches='tight')
        print("  Saved: results/data_preview.png")
    
    return importer

if __name__ == "__main__":
    example_data_import()