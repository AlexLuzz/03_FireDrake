"""
Results visualization
"""
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.interpolate import LinearNDInterpolator, interp1d
from datetime import datetime
import csv


class ResultsPlotter:
    """Creates complete visualization with time series and spatial snapshots"""
    
    def __init__(self, config, mesh):
        self.config = config
        self.mesh = mesh
        self.coords = mesh.coordinates.dat.data

    def plot_complete_results(self, probe_data, snapshots=None, rain_scenario=None, 
                             filename=None, comsol_data_file=None, measured_data_file=None,
                             plot_residuals=False, plot_dates=True,
                             comsol_ref_date=None, measured_ref_date=None,
                             measured_offset=0.0):
        """
        Create complete results figure
        
        Args:
            probe_data: Dictionary from ProbeManager.get_data()
            snapshots: Dictionary from SnapshotManager.snapshots
            rain_scenario: Optional RainScenario for plotting rain events
            filename: Output filename (optional)
            comsol_data_file: Path to CSV with COMSOL modeled data
            measured_data_file: Path to CSV with real measured data
            plot_residuals: Whether to plot COMSOL residuals (auto-enabled if comsol_data_file provided)
            plot_dates: If True and config has time_converter, plot with datetime x-axis (default: True)
            comsol_ref_date: Reference datetime for COMSOL t=0 (e.g., datetime(2024, 3, 3))
            measured_ref_date: Reference datetime for measured data (auto-inferred from config if None)
            measured_offset: Vertical offset to add to measured data in meters (e.g., 0.6 for 60cm)
        """
        # Calculate start_from automatically if we have datetime config
        start_from = 0.0  # Default
        if hasattr(self.config, 'start_datetime') and self.config.start_datetime and comsol_ref_date:
            # Calculate days between comsol_ref_date and simulation start
            start_from = (self.config.start_datetime - comsol_ref_date).total_seconds() / 86400.0
            print(f"ℹ️  Auto-calculated start_from: {start_from:.1f} days "
                  f"(from {comsol_ref_date.strftime('%Y-%m-%d')} to {self.config.start_datetime.strftime('%Y-%m-%d')})")
        
        # Auto-infer measured_ref_date from config if not provided
        if measured_data_file and measured_ref_date is None:
            if hasattr(self.config, 'start_datetime') and self.config.start_datetime:
                measured_ref_date = self.config.start_datetime
                print(f"ℹ️  Using config start_datetime as measured_ref_date: {measured_ref_date.strftime('%Y-%m-%d')}")
        
        # Determine if we can use datetime axis
        use_datetime = plot_dates and hasattr(self.config, 'time_converter') and self.config.time_converter is not None
        
        # Load and align data if provided
        comsol_data = None
        measured_data = None
        
        if comsol_data_file:
            comsol_data = self._load_and_align_data(
                comsol_data_file, 
                start_from, 
                probe_data['times'][-1] / 3600.0 / 24.0,
                data_type='COMSOL',
                ref_date=comsol_ref_date
            )
            # Auto-enable residuals if COMSOL data loaded
            if comsol_data and not plot_residuals:
                plot_residuals = True
        
        if measured_data_file:
            # For measured data, use start_from=0 since it's already aligned to simulation start
            measured_data = self._load_and_align_data(
                measured_data_file,
                start_from=0.0,  # Measured data ref_date is simulation start, so no offset needed
                sim_duration_days=probe_data['times'][-1] / 3600.0 / 24.0,
                data_type='Measured',
                ref_date=measured_ref_date,
                offset=measured_offset
            )
        
        # Calculate layout
        n_rows = 1  # Time series
        if plot_residuals:
            if comsol_data:
                n_rows += 1  # COMSOL residuals
            if measured_data:
                n_rows += 1  # Measured residuals
        if snapshots:
            n_rows += 2  # Snapshots (2 rows)
        
        fig_height = 4 * n_rows
        fig = plt.figure(figsize=(20, fig_height))
        gs = GridSpec(n_rows, 3, figure=fig, hspace=0.25, wspace=0.35)
        
        # Overall title
        title = 'Richards Equation Simulation - Water Table Response'
        if use_datetime and self.config.start_datetime:
            title += f'\n{self.config.start_datetime.strftime("%Y-%m-%d")} to {self.config.time_converter.to_datetime(self.config.t_end).strftime("%Y-%m-%d")}'
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        # Track current row and which is the last time-series row (for x-axis labels)
        current_row = 0
        last_timeseries_row = 0 if not plot_residuals else (1 if comsol_data and not measured_data else 2)
        
        # Plot time series (row 0)
        is_last = (current_row == last_timeseries_row)
        self._plot_time_series(fig, gs[current_row, :], probe_data, rain_scenario, 
                              comsol_data, measured_data, show_xlabel=is_last,
                              use_datetime=use_datetime)
        current_row += 1
        
        # Plot residuals if requested
        if plot_residuals:
            if comsol_data:
                is_last = (current_row == last_timeseries_row)
                self._plot_residuals(fig, gs[current_row, :], probe_data, comsol_data, 
                                    'COMSOL', show_xlabel=is_last,
                                    use_datetime=use_datetime)
                current_row += 1
            
            if measured_data:
                is_last = (current_row == last_timeseries_row)
                self._plot_residuals(fig, gs[current_row, :], probe_data, measured_data,
                                    'Measured', show_xlabel=is_last,
                                    use_datetime=use_datetime)
                current_row += 1
        
        # Plot snapshots (rows after time series)
        if snapshots:
            self._plot_snapshots(fig, gs, snapshots, start_row=current_row,
                               use_datetime=use_datetime)
        
        # Save
        if filename is None:
            filename = self.config.output_dir / f'results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
        
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Plot saved: {filename}")
        plt.close()
    
    def _smooth_data(self, times, data, window_hours=3.0):
        """
        Apply sliding window smoothing to noisy data
        
        Args:
            times: Time array in days
            data: Data array with potential NaN values
            window_hours: Window size in hours for smoothing (default: 1 hour)
        
        Returns:
            Smoothed data array
        """
        # Remove NaN values for smoothing
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return data
        
        times_valid = times[valid_mask]
        data_valid = data[valid_mask]
        
        # Convert window to days
        window_days = window_hours / 24.0
        
        # Apply sliding window average
        smoothed = np.zeros_like(data_valid)
        for i, t in enumerate(times_valid):
            # Find points within window
            window_mask = np.abs(times_valid - t) <= window_days / 2.0
            if np.any(window_mask):
                smoothed[i] = np.mean(data_valid[window_mask])
            else:
                smoothed[i] = data_valid[i]
        
        # Put smoothed data back into full array with NaNs
        result = np.full_like(data, np.nan)
        result[valid_mask] = smoothed
        
        return result
    
    def _load_and_align_data(self, csv_path, start_from, sim_duration_days, data_type='Data', ref_date=None, offset=0.0):
        """
        Load data (COMSOL or measured) and align it with simulation time
        
        Args:
            csv_path: Path to CSV file
            start_from: Time (days) to map to simulation t=0
            sim_duration_days: Simulation duration in days
            data_type: 'COMSOL' or 'Measured' for logging
            ref_date: Reference datetime for t=0 (required if CSV has datetime strings)
            offset: Vertical offset to add to all data values (e.g., 0.6 for 60cm)
        
        Returns:
            Dictionary with aligned time and data arrays, or None if error
        """
        try:
            # Load raw data
            with open(csv_path, 'r', encoding='utf-8-sig') as f:
                first_line = f.readline()
                delimiter = ';' if ';' in first_line else ','
                f.seek(0)
                reader = csv.DictReader(f, delimiter=delimiter)
                rows = [{k.strip(): v.strip() for k, v in row.items()} 
                        for row in reader]
            
            # Find time/date column
            time_col = None
            for k in rows[0].keys():
                if any(term in k.lower() for term in ['time', 'date', 'jour', 'day']):
                    time_col = k
                    break
            
            if not time_col:
                print(f"⚠️  No time/date column found in {csv_path}")
                return None
            
            # Try to parse times - could be floats or datetime strings
            times_raw = []
            is_datetime_col = False
            
            for row in rows:
                time_str = row[time_col]
                try:
                    # Try parsing as float first
                    times_raw.append(float(time_str.replace(',', '.')))
                except ValueError:
                    # Must be a datetime string
                    is_datetime_col = True
                    break
            
            # If datetime strings, need to parse them
            if is_datetime_col:
                if ref_date is None:
                    print(f"⚠️  {data_type} data has datetime strings but no ref_date provided")
                    return None
                
                from datetime import datetime
                times_raw = []
                for row in rows:
                    date_str = row[time_col]
                    # Try common European date formats
                    for fmt in ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d']:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            # Convert to days from ref_date
                            days_diff = (dt - ref_date).total_seconds() / 86400.0
                            times_raw.append(days_diff)
                            break
                        except ValueError:
                            continue
                    else:
                        print(f"⚠️  Could not parse date: {date_str}")
                        return None
                
                times_raw = np.array(times_raw)
                print(f"✓ Parsed {len(times_raw)} datetime strings from {data_type} data")
            else:
                times_raw = np.array(times_raw)
            
            # Parse data columns - look for LTC or Level columns
            data_raw = {}
            column_mapping = {}  # Map original column names to standardized LTC names
            
            for col in rows[0].keys():
                if col.startswith('LTC') or 'ltc' in col.lower() or 'level' in col.lower():
                    try:
                        # Parse values, replacing empty strings with NaN
                        values = []
                        for row in rows:
                            val_str = row[col].replace(',', '.').strip()
                            if val_str == '' or val_str == 'nan':
                                values.append(np.nan)
                            else:
                                values.append(float(val_str))
                        
                        data_raw[col] = np.array(values)
                        
                        # Standardize column names: "Level 101" -> "LTC 101", "Level 102" -> "LTC 102", etc.
                        if 'level' in col.lower():
                            # Extract number from "Level 101 (m)" or "Level 101"
                            import re
                            match = re.search(r'(\d+)', col)
                            if match:
                                num = match.group(1)
                                standardized_name = f"LTC {num}"
                                column_mapping[col] = standardized_name
                        else:
                            column_mapping[col] = col
                            
                    except ValueError as e:
                        print(f"⚠️  Could not parse column {col}: {e}")
                        continue
            
            if not data_raw:
                print(f"⚠️  No LTC/Level data columns found in {csv_path}")
                return None
            
            # Apply column name mapping
            data_raw = {column_mapping[k]: v for k, v in data_raw.items()}
            
            print(f"✓ Loaded {data_type} data: {list(data_raw.keys())}")
            print(f"  Raw time range: {times_raw[0]:.2f} to {times_raw[-1]:.2f} days ({len(times_raw)} points)")
            
            # Filter, shift, and clip
            mask_start = times_raw >= start_from
            if not np.any(mask_start):
                print(f"⚠️  No {data_type} data found at or after start_from={start_from:.2f} days")
                return None
            
            times_filtered = times_raw[mask_start]
            data_filtered = {k: v[mask_start] for k, v in data_raw.items()}
            times_shifted = times_filtered - start_from
            
            mask_clip = times_shifted <= sim_duration_days
            if not np.any(mask_clip):
                print(f"⚠️  No {data_type} data in simulation range after shifting")
                return None
            
            times_aligned = times_shifted[mask_clip]
            data_aligned = {k: v[mask_clip] for k, v in data_filtered.items()}
            
            # Apply smoothing to measured data (it's typically much noisier than COMSOL)
            if data_type == 'Measured':
                print(f"  Applying sliding window smoothing to measured data (1-hour window)...")
                data_aligned = {k: self._smooth_data(times_aligned, v, window_hours=1.0) 
                               for k, v in data_aligned.items()}
            
            # Apply vertical offset if specified
            if offset != 0.0:
                print(f"  Applying vertical offset: {offset:+.2f} m to {data_type} data")
                data_aligned = {k: v + offset for k, v in data_aligned.items()}
            
            # Create final aligned dictionary
            aligned_data = {'times': times_aligned}
            aligned_data.update(data_aligned)
            
            print(f"  Aligned {data_type} data: {len(times_aligned)} points in range [0, {sim_duration_days:.2f}] days")
            
            return aligned_data
            
        except Exception as e:
            print(f"⚠️  Error loading {data_type} data: {e}")
            return None
    
    def _plot_time_series(self, fig, gs_slice, probe_data, rain_scenario, 
                         comsol_data, measured_data, show_xlabel=True, use_datetime=False):
        """Plot water table time series"""
        ax = fig.add_subplot(gs_slice)
        
        # Prepare time data
        times_sec = probe_data['times']
        if use_datetime:
            times_plot = [self.config.time_converter.to_datetime(t) for t in times_sec]
        else:
            times_plot = times_sec / 3600.0  # Convert to hours
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        
        # Plot simulated data (Firedrake)
        for i, (name, data) in enumerate(probe_data['data'].items()):
            # Extract clean probe name (e.g., "LTC 1 (x=8.0m, y=1.0m)" -> "LTC 1")
            clean_name = name.split('(')[0].strip()
            ax.plot(times_plot, data, color=colors[i], linewidth=2.5, 
                   label=f'{clean_name} - Firedrake simulation', marker='o', markersize=3, 
                   markevery=max(1, len(times_plot)//30))
        
        # Determine x-axis limits based on available data
        x_min = times_plot[0]
        x_max = times_plot[-1]
        has_reference_data = False
        
        # Plot COMSOL data if available
        if comsol_data:
            has_reference_data = True
            if use_datetime:
                # COMSOL data times are in days from simulation start
                times_comsol = [self.config.time_converter.to_datetime(t * 86400) for t in comsol_data['times']]
            else:
                times_comsol = comsol_data['times'] * 24  # days to hours
            
            # Update x-limits to where COMSOL data exists
            x_min = max(x_min, times_comsol[0])
            x_max = min(x_max, times_comsol[-1])
            
            for i, ltc_name in enumerate(['LTC 101', 'LTC 102', 'LTC 103']):
                if ltc_name in comsol_data:
                    # Map LTC 101/102/103 to LTC 1/2/3 for consistency
                    display_name = f"LTC {i+1}"
                    ax.plot(times_comsol, comsol_data[ltc_name],
                           color=colors[i], linewidth=2, linestyle='--',
                           label=f'{display_name} - COMSOL model', marker='s', markersize=4,
                           markevery=max(1, len(times_comsol)//20), alpha=0.8)
        
        # Plot measured data if available
        if measured_data:
            has_reference_data = True
            if use_datetime:
                times_measured = [self.config.time_converter.to_datetime(t * 86400) for t in measured_data['times']]
            else:
                times_measured = measured_data['times'] * 24
            
            # Update x-limits to where measured data exists
            x_min = max(x_min, times_measured[0])
            x_max = min(x_max, times_measured[-1])
            
            for i, ltc_name in enumerate(['LTC 101', 'LTC 102', 'LTC 103']):
                if ltc_name in measured_data:
                    # Map LTC 101/102/103 to LTC 1/2/3 for consistency
                    display_name = f"LTC {i+1}"
                    # Filter out NaN values for cleaner plotting
                    valid_mask = ~np.isnan(measured_data[ltc_name])
                    times_valid = [t for t, v in zip(times_measured, valid_mask) if v]
                    data_valid = measured_data[ltc_name][valid_mask]
                    
                    ax.plot(times_valid, data_valid,
                           color=colors[i], linewidth=1.5, linestyle=':',
                           label=f'{display_name} - Field measurements', marker='^', markersize=3,
                           markevery=max(1, len(times_valid)//15), alpha=0.7)
        
        # Rain event shading
        if rain_scenario:
            for idx, event in enumerate(rain_scenario.events):
                if use_datetime:
                    start_dt = self.config.time_converter.to_datetime(event.start_time * 3600)
                    end_dt = self.config.time_converter.to_datetime(event.end_time * 3600)
                    # Only add label for first rain event to avoid duplicates
                    ax.axvspan(start_dt, end_dt, alpha=0.15, color='lightblue', 
                              label='Rain event' if idx == 0 else '')
                else:
                    ax.axvspan(event.start_time, event.end_time, 
                              alpha=0.15, color='lightblue', 
                              label='Rain event' if idx == 0 else '')
        
        ax.set_ylabel('Water Table Elevation (m)', fontsize=12, fontweight='bold')
        if show_xlabel:
            if use_datetime:
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                # Format date axis
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)
        
        ax.grid(True, alpha=0.3)
        
        # Set x-limits to where reference data exists (if any), otherwise full simulation
        if has_reference_data:
            ax.set_xlim([x_min, x_max])
        else:
            if use_datetime:
                ax.set_xlim([times_plot[0], times_plot[-1]])
            else:
                ax.set_xlim([0, times_plot[-1]])
        
        # Remove duplicate labels in legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), 
                 loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=10)
    
    def _plot_residuals(self, fig, gs_slice, probe_data, reference_data, 
                       reference_name, show_xlabel=True, use_datetime=False):
        """Plot residuals: Reference - Firedrake"""
        ax = fig.add_subplot(gs_slice)
        
        # Prepare simulation time data
        times_sec_sim = probe_data['times']
        if use_datetime:
            times_sim = [self.config.time_converter.to_datetime(t) for t in times_sec_sim]
        else:
            times_sim = times_sec_sim / 3600.0
        
        # Prepare reference time data
        if use_datetime:
            times_ref = [self.config.time_converter.to_datetime(t * 86400) for t in reference_data['times']]
        else:
            times_ref = reference_data['times'] * 24
        
        colors = ['#1f77b4', '#2ca02c', '#d62728']
        probe_names = list(probe_data['data'].keys())
        ltc_names = ['LTC 101', 'LTC 102', 'LTC 103']
        
        for i in range(min(len(probe_names), len(ltc_names))):
            ltc_name = ltc_names[i]
            probe_name = probe_names[i]
            
            if ltc_name not in reference_data:
                continue
            
            # Get data
            sim_data = np.array(probe_data['data'][probe_name])
            ref_values = reference_data[ltc_name]
            
            # Interpolate simulation to reference times (need numeric values)
            times_sec_sim_numeric = times_sec_sim
            times_ref_numeric = reference_data['times'] * 86400  # days to seconds
            
            interp_func = interp1d(times_sec_sim_numeric, sim_data, kind='linear', 
                                  bounds_error=False, fill_value='extrapolate')
            sim_interp = interp_func(times_ref_numeric)
            
            # Calculate residuals (Reference - Firedrake)
            residuals = ref_values - sim_interp
            
            # Plot using prepared time arrays
            ax.plot(times_ref, residuals, color=colors[i], linewidth=2.5,
                   marker='o', markersize=3, markevery=max(1, len(times_ref)//30))
            
            # Statistics
            print(f"  {reference_name} - {ltc_name} residuals: mean={residuals.mean():.4f}m, "
                  f"std={residuals.std():.4f}m, max_abs={np.abs(residuals).max():.4f}m")
        
        # Zero line
        ax.axhline(0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
        
        ax.set_ylabel(f'Residual (m)\n[{reference_name} - Firedrake]', 
                     fontsize=11, fontweight='bold')
        if show_xlabel:
            if use_datetime:
                ax.set_xlabel('Date', fontsize=12, fontweight='bold')
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
            else:
                ax.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)
        
        ax.grid(True, alpha=0.3)
        
        # Set x-limits to match reference data extent
        ax.set_xlim([times_ref[0], times_ref[-1]])
    
    def _plot_snapshots(self, fig, gs, snapshots, start_row, use_datetime=False):
        """Plot spatial snapshots (6 plots in 2x3 grid)"""
        sorted_times = sorted(snapshots.keys())[:6]
        while len(sorted_times) < 6:
            sorted_times.append(sorted_times[-1])
        
        x_coords = self.coords[:, 0]
        y_coords = self.coords[:, 1]
        
        for idx, t in enumerate(sorted_times):
            row = start_row + idx // 3
            col = idx % 3
            ax = fig.add_subplot(gs[row, col])
            
            # Get saturation
            sat = snapshots[t]['saturation'].dat.data[:]
            sat = np.clip(sat, 0, 1)
            
            # Interpolate to regular grid
            xi = np.linspace(0, self.config.Lx, 200)
            yi = np.linspace(0, self.config.Ly, 100)
            Xi, Yi = np.meshgrid(xi, yi)
            
            interp = LinearNDInterpolator(np.column_stack((x_coords, y_coords)), sat)
            Zi = interp(Xi, Yi)
            
            # Plot
            cf = ax.contourf(Xi, Yi, Zi, levels=np.linspace(0, 1, 25),
                            cmap='Blues', vmin=0, vmax=1)
            
            if idx == 0:
                contour_for_cbar = cf
            
            # Monitoring points
            if hasattr(self.config, 'probes_positions'):
                colors = ['#1f77b4', '#2ca02c', '#d62728']
                for i, (x, y) in enumerate(self.config.probes_positions):
                    ax.plot(x, y, '*', color=colors[i], 
                           markersize=12, markeredgecolor='black', markeredgewidth=0.8)
            
            ax.set_xlabel('x (m)', fontsize=10)
            ax.set_ylabel('y (m)', fontsize=10)
            
            # Title with time or datetime
            if use_datetime:
                dt_title = self.config.time_converter.to_datetime(t)
                ax.set_title(f'{dt_title.strftime("%Y-%m-%d %H:%M")}', 
                           fontsize=11, fontweight='bold')
            else:
                ax.set_title(f't = {t/3600:.1f}h', fontsize=11, fontweight='bold')
            
            ax.set_aspect('equal')
            ax.set_xlim(0, self.config.Lx)
            ax.set_ylim(0, self.config.Ly)
        
        # Colorbar
        cbar_ax = fig.add_axes([0.92, 0.11, 0.02, 0.56])
        cbar = fig.colorbar(contour_for_cbar, cax=cbar_ax)
        cbar.set_label('Saturation', fontsize=12, fontweight='bold')