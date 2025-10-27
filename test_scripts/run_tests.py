"""
Simple test runner for FireDrake simulation package.

This script provides a convenient way to run different types of tests:
- Unit tests for individual components
- Integration tests for complete workflows
- Performance benchmarks

Usage:
    python run_tests.py [test_type]
    
    test_type options:
    - basic: Run basic import and functionality tests
    - richards: Run full Richards equation simulation test
    - all: Run all available tests (default)
"""

import sys
import os
import argparse
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def run_basic_tests():
    """Run basic package tests."""
    print("="*50)
    print("RUNNING BASIC TESTS")
    print("="*50)
    
    try:
        from test_main_richards import run_basic_tests
        return run_basic_tests()
    except Exception as e:
        print(f"Error running basic tests: {e}")
        return False


def run_richards_test():
    """Run the Richards equation simulation test."""
    print("="*50)
    print("RUNNING RICHARDS SIMULATION TEST")
    print("="*50)
    
    try:
        from test_main_richards import test_richards_simulation
        return test_richards_simulation()
    except Exception as e:
        print(f"Error running Richards test: {e}")
        return False


def run_all_tests():
    """Run all available tests."""
    print("Running all tests...")
    
    tests = [
        ("Basic Tests", run_basic_tests),
        ("Richards Simulation Test", run_richards_test),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        print(f"Starting: {test_name}")
        print(f"{'='*60}")
        
        try:
            start_time = datetime.now()
            success = test_func()
            end_time = datetime.now()
            duration = end_time - start_time
            
            results[test_name] = {
                'success': success,
                'duration': duration,
                'error': None
            }
            
        except Exception as e:
            results[test_name] = {
                'success': False,
                'duration': None,
                'error': str(e)
            }
    
    # Print summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r['success'])
    
    for test_name, result in results.items():
        status = "PASS" if result['success'] else "FAIL"
        duration_str = f"({result['duration'].total_seconds():.1f}s)" if result['duration'] else ""
        
        print(f"{test_name:.<40} {status} {duration_str}")
        
        if not result['success'] and result['error']:
            print(f"  Error: {result['error']}")
    
    print(f"\nResults: {passed_tests}/{total_tests} tests passed")
    
    return passed_tests == total_tests


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(description='FireDrake Test Runner')
    parser.add_argument(
        'test_type', 
        nargs='?', 
        default='all',
        choices=['basic', 'richards', 'all'],
        help='Type of tests to run (default: all)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    print("FireDrake Test Runner")
    print("====================")
    print(f"Test type: {args.test_type}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Set up test environment
    if args.verbose:
        print("Setting up test environment...")
    
    # Run requested tests
    if args.test_type == 'basic':
        success = run_basic_tests()
    elif args.test_type == 'richards':
        success = run_richards_test()
    elif args.test_type == 'all':
        success = run_all_tests()
    else:
        print(f"Unknown test type: {args.test_type}")
        return 1
    
    # Return appropriate exit code
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)