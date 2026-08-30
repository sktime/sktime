#!/usr/bin/env python3
"""
sktime Anomaly Detector Stress Test
====================================

This script tests detectors with pathological inputs to discover error handling gaps:
- NaN/missing values
- Zero-variance (constant) arrays
- Dimension mismatches (1D vs 2D)
- Empty data
- Extreme values (inf, -inf)
- Single-sample data

These tests help identify:
1. Missing input validation
2. Improper error messages
3. Unhandled edge cases in algorithms

Usage:
    python stress_test.py                  # Test all detectors with all stress tests
    python stress_test.py --quiet          # Minimal output
    python stress_test.py --verbose        # Show full tracebacks
    python stress_test.py --detector SubLOF # Test only specific detector
"""

import sys
import traceback
from collections import defaultdict
import warnings

try:
    import numpy as np
    import pandas as pd
    from sktime.registry import all_estimators
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please run: pip install -e .[dev,all_extras,detection]")
    sys.exit(1)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


class StressTestSuite:
    """Edge-case stress testing for sktime detectors."""

    def __init__(self, verbose=False, quiet=False, specific_detector=None):
        self.verbose = verbose
        self.quiet = quiet
        self.specific_detector = specific_detector
        self.results = defaultdict(lambda: defaultdict(list))
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    @staticmethod
    def create_nan_data():
        """Create data with NaN values."""
        return pd.DataFrame({
            "col_0": [1.0, 2.0, np.nan, 4.0, 5.0],
            "col_1": [2.0, np.nan, 3.0, 5.0, 6.0],
            "col_2": [3.0, 4.0, 5.0, np.nan, 7.0],
        })

    @staticmethod
    def create_zero_variance_data():
        """Create constant array (zero variance)."""
        return pd.DataFrame({
            "col_0": [0.0] * 10,
            "col_1": [0.0] * 10,
            "col_2": [0.0] * 10,
        })

    @staticmethod
    def create_normal_data():
        """Create normal test data."""
        return pd.DataFrame({
            "col_0": np.linspace(1, 10, 10),
            "col_1": np.linspace(2, 11, 10),
            "col_2": np.linspace(3, 12, 10),
        })

    @staticmethod
    def create_normal_data_1d():
        """Create 1D Series (incorrect shape)."""
        return pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    @staticmethod
    def create_empty_data():
        """Create empty DataFrame."""
        return pd.DataFrame()

    @staticmethod
    def create_single_sample_data():
        """Create single-sample data."""
        return pd.DataFrame({
            "col_0": [1.0],
            "col_1": [2.0],
            "col_2": [3.0],
        })

    @staticmethod
    def create_extreme_values_data():
        """Create data with inf and -inf."""
        return pd.DataFrame({
            "col_0": [1.0, np.inf, 3.0, 4.0],
            "col_1": [2.0, 3.0, -np.inf, 5.0],
            "col_2": [3.0, 4.0, 5.0, 6.0],
        })

    @staticmethod
    def get_test_instances(detector_cls):
        """Get valid test instances using sktime's own factory method."""
        try:
            instances, _ = detector_cls.create_test_instances_and_names()
            return instances if instances else [detector_cls()]
        except Exception:
            # Fallback: try bare instantiation
            try:
                return [detector_cls()]
            except Exception:
                return []

    def run_stress_test(self, detector_name, detector_cls):
        """Run all stress tests on a detector."""
        if self.specific_detector and detector_name != self.specific_detector:
            return

        if not self.quiet:
            print(f"\nTesting: {detector_name}")
            print("-" * 60)

        # Check if detector has soft dependencies that aren't installed
        try:
            tags = detector_cls.get_class_tags()
            deps = tags.get("python_dependencies")
            if deps:
                from skbase.utils.dependencies import _check_soft_dependencies
                if not _check_soft_dependencies(deps, severity="none"):
                    if not self.quiet:
                        print(f"  ⊘ SKIPPED - missing soft deps: {deps}")
                    return
        except Exception:
            pass

        # Get properly initialised test instances
        instances = self.get_test_instances(detector_cls)
        if not instances:
            if not self.quiet:
                print(f"  ⊘ SKIPPED - could not instantiate")
            return

        # Use first test instance as representative
        detector_instance = instances[0]

        # Define stress tests: (name, data_fn, expected_behavior)
        stress_tests = [
            (
                "NaN Values",
                self.create_nan_data,
                "Handle gracefully or raise ValueError",
            ),
            (
                "Zero Variance",
                self.create_zero_variance_data,
                "Handle or raise informative error",
            ),
            (
                "1D Series Input",
                self.create_normal_data_1d,
                "Raise shape error, not math exception",
            ),
            (
                "Empty Data",
                self.create_empty_data,
                "Raise error about empty data",
            ),
            (
                "Single Sample",
                self.create_single_sample_data,
                "Handle or raise informative error",
            ),
            (
                "Extreme Values (inf)",
                self.create_extreme_values_data,
                "Handle or raise informative error",
            ),
        ]

        for test_name, data_fn, expected_behavior in stress_tests:
            self.total_tests += 1
            result = self._run_single_test(
                detector_name, detector_cls, test_name, data_fn, expected_behavior
            )

            if result["passed"]:
                self.passed_tests += 1
                status = "✅"
            else:
                self.failed_tests += 1
                status = "❌"

            if not self.quiet:
                print(f"  {status} {test_name:20s} - {result['status']}")

            self.results[detector_name][test_name] = result

    def _run_single_test(
        self, detector_name, detector_cls, test_name, data_fn, expected_behavior
    ):
        """Execute a single stress test."""
        try:
            # Create test data
            X = data_fn()

            # Get a fresh properly-initialised instance each time
            instances = self.get_test_instances(detector_cls)
            detector = instances[0]

            # Try to fit
            detector.fit(X)

            # If we get here without exception, test passed
            return {
                "passed": True,
                "status": "Handled gracefully",
                "error": None,
                "error_type": None,
            }

        except ValueError as e:
            # ValueError is acceptable (informative error)
            error_msg = str(e)[:80]
            return {
                "passed": True,
                "status": f"ValueError (good): {error_msg}",
                "error": str(e),
                "error_type": "ValueError",
            }

        except TypeError as e:
            # TypeError might be acceptable for shape errors
            error_msg = str(e)[:80]
            return {
                "passed": True,
                "status": f"TypeError (acceptable): {error_msg}",
                "error": str(e),
                "error_type": "TypeError",
            }

        except RuntimeError as e:
            # RuntimeError from sklearn might be acceptable
            error_msg = str(e)[:80]
            return {
                "passed": True,
                "status": f"RuntimeError (acceptable): {error_msg}",
                "error": str(e),
                "error_type": "RuntimeError",
            }

        except Exception as e:
            # Other exceptions are failures
            error_type = type(e).__name__
            error_msg = str(e)[:80]
            return {
                "passed": False,
                "status": f"❌ {error_type}: {error_msg}",
                "error": str(e) if self.verbose else error_msg,
                "error_type": error_type,
                "traceback": traceback.format_exc() if self.verbose else None,
            }

    def run(self):
        """Run stress tests on all detectors."""
        print("\n" + "=" * 60)
        print("sktime ANOMALY DETECTOR STRESS TEST SUITE")
        print("=" * 60)
        print()

        # Get detectors
        print("Fetching anomaly detectors from registry...")
        detectors = all_estimators(
            estimator_types="detector",
            filter_tags={"task": "anomaly_detection"},
            return_names=True,
        )

        if not detectors:
            print("⚠ No anomaly detectors found!")
            return 1

        print(f"✓ Found {len(detectors)} detectors\n")

        # Run stress tests
        for name, detector_cls in detectors:
            self.run_stress_test(name, detector_cls)

        # Print summary
        self._print_summary()

        return 0 if self.failed_tests == 0 else 1

    def _print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 60)
        print("STRESS TEST SUMMARY")
        print("=" * 60)
        print(f"Total tests run:        {self.total_tests}")
        print(f"Passed (handled well):  {self.passed_tests}")
        print(f"Failed (bad errors):    {self.failed_tests}")

        if self.failed_tests > 0:
            print("\n" + "=" * 60)
            print("DETECTORS WITH FAILURES")
            print("=" * 60)

            for detector_name in sorted(self.results.keys()):
                failures = [
                    (test_name, result)
                    for test_name, result in self.results[detector_name].items()
                    if not result["passed"]
                ]

                if failures:
                    print(f"\n{detector_name}:")
                    for test_name, result in failures:
                        print(f"  - {test_name}: {result['error_type']}")
                        if self.verbose and result.get("traceback"):
                            print(f"    {result['traceback'][:200]}")

        print("\n" + "=" * 60)


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="sktime Anomaly Detector Stress Test Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python stress_test.py               # Test all detectors
  python stress_test.py --verbose     # Show full tracebacks
  python stress_test.py --quiet       # Minimal output
  python stress_test.py --detector SubLOF  # Test single detector
        """,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show full error tracebacks",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Minimal output",
    )
    parser.add_argument(
        "-d", "--detector",
        type=str,
        help="Test specific detector by name",
    )

    args = parser.parse_args()

    tester = StressTestSuite(
        verbose=args.verbose,
        quiet=args.quiet,
        specific_detector=args.detector,
    )

    exit_code = tester.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
