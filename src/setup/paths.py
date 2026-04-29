from pathlib import Path

class ProjectPaths:
    """Centralized path management for the project"""
    def __init__(self, user='AQ96560', project_name=None):
        """
        Initialize project paths based on the user.
        
        Parameters
        ----------
        user : str
            Username for path configuration (default: 'AQ96560', can be 'alexi' when used from home computer)
        project_name : str, optional
            Name for a specific project folder. If provided, creates a main folder with DATA and OUTPUT subfolders.
            If None, uses the default project structure.
        """
        self.USER = user
        self.project_name = project_name
        self.base_dir = Path(f'/mnt/c/Users/{user}/OneDrive - ETS/02 - Alexis Luzy/01_Modelization')
        
        self.DATA_DIR = self.base_dir / 'DATA' 
        
        if project_name:
            self.OUTPUT_DIR = self.base_dir / 'PROJECTS' / project_name
        else:
            self.OUTPUT_DIR = self.base_dir / 'OUTPUT'
            
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # Data file paths
        self.RAF_COMSOL_PZ_CG = self.DATA_DIR / 'RAF_COMSOL_PZ_CG.csv'
        self.RAF_METEO = self.DATA_DIR / 'RAF_METEO.csv'
        self.MEASURED_PZ_CG = self.DATA_DIR / 'MEASURED_PZ_CG.csv'

    def __repr__(self):
        if self.project_name:
            return f"ProjectPaths(project='{self.project_name}', base={self.base_dir})"
        return f"ProjectPaths(base={self.base_dir})"