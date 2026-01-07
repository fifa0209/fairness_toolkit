"""
Master test runner for Fairness Toolkit.
Runs all module tests in sequence and generates a comprehensive report.

Usage:
    python run_all_tests.py                    # Run all tests
    python run_all_tests.py --module shared    # Test specific module
    python run_all_tests.py --quick            # Run quick tests only
    python run_all_tests.py --verbose          # Detailed output
"""

import sys
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import time


class TestRunner:
    """Orchestrates testing across all modules."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = {}
        self.start_time = None
        
    def run_test(self, test_file: str, module_name: str) -> Tuple[bool, str, float]:
        """Run a single test file and capture results."""
        print(f"\n{'='*70}")
        print(f"Testing {module_name.upper()}")
        print(f"{'='*70}")
        
        if not Path(test_file).exists():
            return False, f"Test file not found: {test_file}", 0.0
        
        start = time.time()
        
        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=not self.verbose,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            duration = time.time() - start
            
            success = result.returncode == 0
            output = result.stdout if not self.verbose else ""
            
            if success:
                print(f"✅ {module_name} tests PASSED ({duration:.1f}s)")
            else:
                print(f"❌ {module_name} tests FAILED ({duration:.1f}s)")
                if result.stderr and not self.verbose:
                    print(f"\nError output:\n{result.stderr[:500]}")
            
            return success, output, duration
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start
            print(f"⏱️  {module_name} tests TIMEOUT ({duration:.1f}s)")
            return False, "Test timeout", duration
            
        except Exception as e:
            duration = time.time() - start
            print(f"❌ {module_name} tests ERROR: {e}")
            return False, str(e), duration
    
    def run_all_tests(self, test_suite: Dict[str, str]) -> Dict[str, Dict]:
        """Run all tests in the suite."""
        self.start_time = datetime.now()
        
        print("╔" + "="*68 + "╗")
        print("║" + " "*15 + "FAIRNESS TOOLKIT TEST SUITE" + " "*26 + "║")
        print("╚" + "="*68 + "╝")
        print(f"\nStarted: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Running {len(test_suite)} test modules...\n")
        
        for module_name, test_file in test_suite.items():
            success, output, duration = self.run_test(test_file, module_name)
            
            self.results[module_name] = {
                'success': success,
                'output': output,
                'duration': duration,
                'test_file': test_file
            }
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate a test report."""
        end_time = datetime.now()
        total_duration = (end_time - self.start_time).total_seconds()
        
        passed = sum(1 for r in self.results.values() if r['success'])
        failed = len(self.results) - passed
        
        report = []
        report.append("\n" + "="*70)
        report.append("TEST SUMMARY")
        report.append("="*70)
        report.append(f"\nTotal Tests: {len(self.results)}")
        report.append(f"Passed: {passed} ✅")
        report.append(f"Failed: {failed} ❌")
        report.append(f"Success Rate: {passed/len(self.results)*100:.1f}%")
        report.append(f"Total Duration: {total_duration:.1f}s")
        
        report.append("\n" + "-"*70)
        report.append("DETAILED RESULTS")
        report.append("-"*70)
        
        for module_name, result in self.results.items():
            status = "✅ PASSED" if result['success'] else "❌ FAILED"
            report.append(f"\n{module_name:20} {status:15} ({result['duration']:.1f}s)")
        
        if failed > 0:
            report.append("\n" + "-"*70)
            report.append("FAILED TESTS")
            report.append("-"*70)
            for module_name, result in self.results.items():
                if not result['success']:
                    report.append(f"\n{module_name}:")
                    report.append(f"  File: {result['test_file']}")
                    if result['output']:
                        report.append(f"  Error: {result['output'][:200]}")
        
        report.append("\n" + "="*70)
        report.append(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "test_report.txt"):
        """Save report to file."""
        report = self.generate_report()
        
        output_dir = Path("test_results")
        output_dir.mkdir(exist_ok=True)
        
        filepath = output_dir / filename
        filepath.write_text(report)
        
        print(f"\n📄 Report saved to: {filepath}")
        return filepath


def main():
    parser = argparse.ArgumentParser(description="Run Fairness Toolkit tests")
    parser.add_argument(
        '--module',
        choices=['shared', 'measurement', 'pipeline', 'training', 'monitoring', 'all'],
        default='all',
        help="Which module to test"
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help="Run quick tests only (skip long-running tests)"
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help="Show detailed output"
    )
    parser.add_argument(
        '--save-report',
        action='store_true',
        help="Save test report to file"
    )
    
    args = parser.parse_args()
    
    # Define test suite
    all_tests = {
        'shared': 'test_shared_modules.py',
        'measurement': 'test_measurement_module.py',
        'pipeline': 'test_pipeline_module.py',
        'training': 'test_training_module.py',
        'monitoring': 'test_monitoring_module.py',
    }
    
    # Filter based on module selection
    if args.module == 'all':
        test_suite = all_tests
    else:
        test_suite = {args.module: all_tests[args.module]}
    
    # Run tests
    runner = TestRunner(verbose=args.verbose)
    results = runner.run_all_tests(test_suite)
    
    # Generate and display report
    report = runner.generate_report()
    print(report)
    
    # Save report if requested
    if args.save_report:
        runner.save_report(
            filename=f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
    
    # Exit with appropriate code
    all_passed = all(r['success'] for r in results.values())
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()