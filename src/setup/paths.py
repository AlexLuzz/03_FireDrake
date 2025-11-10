from pathlib import Path
import os

class ProjectPaths:
    """Centralized path management for the project"""
    
    def __init__(self, user='AQ96560', project_name=None, base_dir=None):
        """
        Initialize project paths based on the user.
        
        Parameters
        ----------
        user : str
            Username for path configuration (default: 'AQ96560', can be 'alexi' when used from home computer)
        project_name : str, optional
            Name for a specific project folder. If provided, creates a main folder with DATA and OUTPUT subfolders.
            If None, uses the default project structure.
        base_dir : Path or str, optional
            Base directory for the project. If None, auto-detects based on environment:
            - WSL: Uses /mnt/c/Users/{user}/OneDrive - ETS/02 - Alexis Luzy/01_Modelization
            - Windows: Uses C:/Users/{user}/OneDrive - ETS/02 - Alexis Luzy/01_Modelization
            - Linux: Uses /home/{user}/03_FireDrake
        """
        self.USER = user
        self.project_name = project_name
        
        # Auto-detect environment and set base directory
        if base_dir is None:
            base_dir = self._detect_base_dir()
        
        # Convert to Path
        self.base_dir = Path(base_dir)
        
        # Custom project folder setup
        if project_name:
            self.CUSTOM_PROJECT_ROOT = self.base_dir / 'PROJECTS' / project_name
            self.DATA_DIR = self.base_dir / 'DATA'
            self.OUTPUT_DIR = self.CUSTOM_PROJECT_ROOT / 'OUTPUT'
        else:
            # Default structure (backward compatibility)
            self.CUSTOM_PROJECT_ROOT = None
            self.DATA_DIR = self.base_dir / 'DATA'
            self.OUTPUT_DIR = self.base_dir / 'OUTPUT'
        
        # Data file paths
        self.RAF_COMSOL_PZ_CG = self.DATA_DIR / 'RAF_COMSOL_PZ_CG.csv'
        self.RAF_METEO = self.DATA_DIR / 'RAF_METEO.csv'
        self.MEASURED_PZ_CG = self.DATA_DIR / 'MEASURED_PZ_CG.csv'

        # Create directories if they don't exist
        self._create_directories()
    
    def _detect_base_dir(self):
        """Auto-detect base directory based on environment"""
        # Check if running in WSL
        if os.path.exists('/proc/version'):
            try:
                with open('/proc/version', 'r') as f:
                    if 'microsoft' in f.read().lower():
                        # Running in WSL - use Windows filesystem via /mnt/c/
                        wsl_windows_path = Path(f'/mnt/c/Users/{self.USER}/OneDrive - ETS/02 - Alexis Luzy/01_Modelization')
                        if wsl_windows_path.exists():
                            print(f"🐧 WSL detected - using Windows filesystem: {wsl_windows_path}")
                            return wsl_windows_path
                        else:
                            print(f"⚠️  WSL detected but {wsl_windows_path} doesn't exist")
                            # Fallback to Linux home
                            linux_fallback = Path(f'/home/{self.USER}/03_FireDrake')
                            print(f"   Using Linux home: {linux_fallback}")
                            return linux_fallback
            except Exception as e:
                print(f"⚠️  Error detecting WSL: {e}")
        
    def _create_directories(self):
        """Create necessary directories if they don't exist"""
        dirs = [
            self.DATA_DIR,
            self.OUTPUT_DIR,
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Log project creation
        if self.project_name:
            print(f"✅ Created project: '{self.project_name}'")
            print(f"   📂 Root: {self.CUSTOM_PROJECT_ROOT}")
            print(f"   📂 Data: {self.DATA_DIR}")
            print(f"   📂 Output: {self.OUTPUT_DIR}")
        else:
            print(f"✅ Using base directory: {self.base_dir}")
            print(f"   📂 Data: {self.DATA_DIR}")
            print(f"   📂 Output: {self.OUTPUT_DIR}")
    
    def __repr__(self):
        if self.project_name:
            return f"ProjectPaths(project='{self.project_name}', base={self.base_dir})"
        return f"ProjectPaths(base={self.base_dir})"