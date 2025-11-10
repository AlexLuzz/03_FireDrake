def print_custom_report(self, config, domain, plotter, custom_pages, filename=None):
        """
        Generate custom report with user-defined pages
        
        custom_pages format:
        {
            'comparison': {
                'type': 'timeseries_comparison',
                'sim_data': {...},
                'analytical_data': {...},
                'title': 'Simulated vs Analytical',
                'ylabel': 'Concentration (kg/m³)'
            },
            'residuals': {
                'type': 'timeseries',
                'data': {...},
                'title': 'Residuals',
                'ylabel': 'Residual (kg/m³)'
            },
            'snapshots': {
                'type': 'snapshots',
                'data': {...},
                'title': 'Concentration Field'
            }
        }
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"custom_report_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        print(f"\n📄 Generating custom report: {filepath}")
        
        with PdfPages(str(filepath)) as pdf:
            for page_name, page_config in custom_pages.items():
                print(f"  - {page_name}")
                
                page_type = page_config.get('type')
                
                # Route to appropriate page creator based on type
                if page_type == 'timeseries_comparison':
                    fig = self._create_comparison_timeseries_page(page_config)
                elif page_type == 'timeseries':
                    fig = self._create_custom_timeseries_page(page_config)
                elif page_type == 'snapshots':
                    fig = self._create_custom_snapshots_page(page_config)
                else:
                    print(f"    Warning: Unknown page type '{page_type}'")
                    continue
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            
            # PDF metadata
            d = pdf.infodict()
            d['Title'] = 'Custom Simulation Report'
            d['Author'] = 'Firedrake Simulation'
            d['Subject'] = f"Simulation: {config.project_name}"
            d['CreationDate'] = datetime.now()
        
        print(f"✓ Report saved: {filepath}")
        return filepath


    
    

    
    # ============================================================================
    # CUSTOM PAGE CREATORS - For verification/comparison reports
    # ============================================================================
    
    def _create_comparison_timeseries_page(self, config):
        """Plot simulated vs analytical time series on same axes"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        sim_data = config['sim_data']
        analytical_data = config['analytical_data']
        times_hours = np.array(sim_data['times']) / 3600.0
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Plot each probe: solid line = simulated, dashed = analytical
        for i, probe_name in enumerate([k for k in sim_data.keys() if k != 'times']):
            color = colors[i % len(colors)]
            ax.plot(times_hours, sim_data[probe_name], color=color, linestyle='-',
                   linewidth=2.5, label=f'{probe_name} (Simulated)', alpha=0.8)
            ax.plot(times_hours, analytical_data[probe_name], color=color, linestyle='--',
                   linewidth=2, label=f'{probe_name} (Analytical)', alpha=0.8)
        
        ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax.set_ylabel(config.get('ylabel', 'Value'), fontweight='bold', fontsize=12)
        ax.set_title(config['title'], fontweight='bold', fontsize=14)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        if len(times_hours) > 0:
            ax.set_xlim(times_hours[0], times_hours[-1])
        
        plt.tight_layout()
        return fig
    
    def _create_custom_timeseries_page(self, config):
        """Plot custom time series data (e.g., residuals)"""
        fig, ax = plt.subplots(figsize=(11, 8.5))
        
        data = config['data']
        times_hours = np.array(data['times']) / 3600.0
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        # Plot each series
        for i, probe_name in enumerate([k for k in data.keys() if k != 'times']):
            ax.plot(times_hours, data[probe_name], color=colors[i % len(colors)],
                   linestyle='-', linewidth=2, label=probe_name, alpha=0.8)
        
        ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax.set_ylabel(config.get('ylabel', 'Value'), fontweight='bold', fontsize=12)
        ax.set_title(config['title'], fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if len(times_hours) > 0:
            ax.set_xlim(times_hours[0], times_hours[-1])
        
        plt.tight_layout()
        return fig
    
    def _create_custom_snapshots_page(self, config):
        """Plot custom spatial snapshots"""
        from firedrake import tripcolor
        
        snapshots = config['data']
        snapshot_times = sorted(list(snapshots.keys()))
        n_snapshots = len(snapshot_times)
        
        if n_snapshots == 0:
            return self._create_no_snapshots_placeholder(config['title'])
        
        # Arrange in grid (max 3 columns)
        cols = min(3, n_snapshots)
        rows = (n_snapshots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(11, 8.5))
        if n_snapshots == 1:
            axes = [axes]
        elif rows == 1:
            axes = list(axes)
        else:
            axes = axes.flatten()
        
        # Plot each snapshot
        for i, t in enumerate(snapshot_times):
            if i < len(axes):
                try:
                    tripcolor(snapshots[t]['concentration'], axes=axes[i])
                    axes[i].set_title(f't = {t/3600:.1f}h', fontweight='bold')
                    axes[i].set_aspect('equal')
                except Exception:
                    axes[i].text(0.5, 0.5, 'Plot error', ha='center', va='center')
                    axes[i].set_xlim(0, 1)
                    axes[i].set_ylim(0, 1)
        
        # Hide unused subplots
        for i in range(n_snapshots, len(axes)):
            axes[i].axis('off')
        
        fig.suptitle(config['title'], fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        return fig
    
    