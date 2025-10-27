# Test Scripts

This folder contains test scripts and examples for the FireDrake simulation package.

## Available Tests

### 1. `test_main_richards.py`
A comprehensive test of the Richards equation simulation workflow. This script:
- Tests all major components of the simulation pipeline
- Demonstrates proper import usage with the new package structure
- Provides a template for creating new simulation scripts
- Includes error handling and debugging information

**Usage:**
```bash
python test_main_richards.py
```

### 2. `run_tests.py`
A test runner that can execute different types of tests:

**Usage:**
```bash
# Run all tests
python run_tests.py

# Run only basic tests
python run_tests.py basic

# Run only Richards simulation test
python run_tests.py richards

# Run with verbose output
python run_tests.py --verbose
```

## Creating New Tests

When creating new test scripts:

1. **Import Structure**: Use the proper import structure shown in `test_main_richards.py`
2. **Error Handling**: Include try-except blocks for robust testing
3. **Documentation**: Add clear docstrings explaining what the test does
4. **Modularity**: Break tests into functions for easier maintenance

### Import Template

```python
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from package modules
from src.solver import RichardsSolver, BoundaryConditionManager
from src.physics import Domain, MaterialField, till, terreau
from src.setup import SimulationConfig, ProbeManager, SnapshotManager
from src.visualization import ResultsPlotter
```

## Test Categories

### Basic Tests
- Import verification
- Basic object creation
- Simple functionality checks

### Integration Tests
- Complete simulation workflows
- Module interaction testing
- End-to-end functionality

### Performance Tests
- Timing benchmarks
- Memory usage monitoring
- Scaling analysis

## Output

Test results and generated files are saved to:
- `../results/` - Simulation outputs
- `../data_output/` - Processed data
- Console output with detailed progress information

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure you're running from the `test_scripts` directory or using the provided path setup
2. **Missing Dependencies**: Check that FireDrake and all required packages are installed
3. **Data Files**: Some tests may require input data files in the `data_input/` directory

### Getting Help

- Check the console output for detailed error messages
- Use the verbose flag (`-v`) for more information
- Review the main package documentation
- Examine working examples in the main directory

## Contributing

When adding new tests:
1. Follow the naming convention: `test_[component]_[description].py`
2. Update this README with test descriptions
3. Add the test to `run_tests.py` if it should be part of the test suite
4. Include proper error handling and cleanup