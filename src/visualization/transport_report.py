

    
    
    def _create_transport_setup_page(self, config, domain, contaminant_props):
        """Create setup page: config + contaminant + materials + geometry"""
        fig = plt.figure(figsize=(11, 14))
        gs = gridspec.GridSpec(5, 3, height_ratios=[0.8, 0.5, 1, 0.1, 1],
                              width_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        
        # Top: Configuration text
        self._add_config_text(fig, gs[0, :], config, domain)
        
        # Contaminant properties text
        if contaminant_props:
            self._add_contaminant_text(fig, gs[1, :], contaminant_props)
        
        # Material curves (3 subplots - more compact)
        if hasattr(domain, 'materials'):
            self._add_material_curves(fig, gs, row=2, materials=domain.materials, n_plots=3)
        
        # Domain geometry
        self._add_domain_geometry(fig, gs, row=4, domain=domain)
        
        fig.suptitle('COUPLED FLOW-TRANSPORT SIMULATION - MODEL SETUP',
                    fontsize=16, fontweight='bold', y=0.98)
        return fig

    def print_transport_report(self, config, domain, plotter, contaminant_props=None, filename=None):
        """Generate transport report: setup + timeseries + snapshots"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transport_report_{timestamp}.pdf"
        
        filepath = self.output_dir / filename
        print(f"\n📄 Generating Transport report: {filepath}")
        
        with PdfPages(str(filepath)) as pdf:
            # Page 1: Configuration + Contaminant + Materials + Domain
            print("  - Setup page")
            fig = self._create_transport_setup_page(config, domain, contaminant_props)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 2: Concentration time series
            print("  - Time series page")
            fig = self._create_timeseries_page(plotter, 'concentration', compare=False)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            
            # Page 3: Concentration snapshots
            print("  - Snapshots page")
            fig = self._create_snapshots_page(plotter, 'concentration')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # Page 4: Mass loss time series
            # Optional: Mass loss / mass change page (if available)
            print("  - Mass loss page")
            fig = self._create_mass_loss_page(plotter, 'mass_loss')
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

            # PDF metadata
            d = pdf.infodict()
            d['Title'] = 'Coupled Flow-Transport Simulation Report'
            d['Author'] = 'Firedrake Simulation'
            d['Subject'] = f"Simulation: {config.project_name}"
            d['CreationDate'] = datetime.now()
        
        print(f"✓ Report saved: {filepath}")
        return filepath