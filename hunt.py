#!/usr/bin/env python3
"""
sktime Anomaly Detector API Compliance Hunter
==============================================

This script runs sktime's built-in check_estimator() on all anomaly detectors
to discover API compliance violations and edge-case bugs.

It systematically tests ~50 API requirements per detector and reports:
- Which detectors pass all checks
- Which detectors fail specific checks
- Error messages for failed checks (useful for identifying root causes)

Usage:
    python hunt.py                    # Test all detectors
    python hunt.py --verbose          # Detailed output per check
    python hunt.py --fail-fast        # Stop on first failure
    python hunt.py --exclude SubLOF   # Skip specific detectors
"""

import sys
import traceback
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    from sktime.registry import all_estimators
    from sktime.utils.estimator_checks import check_estimator
except ImportError as e:
    print(f"Error: Required module not found: {e}")
    print("Please run: pip install -e .[dev,all_extras,detection]")
    sys.exit(1)


class DetectorHunter:
    """Automated API compliance tester for sktime detectors."""

    def __init__(self, verbose=False, fail_fast=False, exclude=None):
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.exclude = exclude or []
        self.results = defaultdict(dict)
        self.passed = []
        self.failed = []

    def get_detectors(self):
        """Fetch all anomaly detectors from registry."""
        print("Fetching anomaly detectors from registry...")
        detectors = all_estimators(
            estimator_types="detector",
            filter_tags={"task": "anomaly_detection"},
            return_names=True,
        )

        if not detectors:
            print("⚠ No anomaly detectors found!")
            print("  Try: pip install sktime[detection]")
            return []

        print(f"✓ Found {len(detectors)} anomaly detectors\n")
        return detectors

    def test_detector(self, name, detector_cls):
        """Run check_estimator on a single detector."""
        if name in self.exclude:
            print(f"⊘ SKIPPED: {name:30s} (excluded)")
            return

        print(f"Testing: {name:30s} ", end="", flush=True)

        try:
            # Run check_estimator with raise_exceptions=False to capture all errors
            results = check_estimator(
                detector_cls,
                raise_exceptions=False,
                verbose=0,  # Silent output from check_estimator
            )

            # Analyze results
            passed_checks = []
            failed_checks = []

            for check_name, result in results.items():
                if isinstance(result, Exception):
                    failed_checks.append((check_name, result))
                else:
                    passed_checks.append(check_name)

            if not failed_checks:
                print("✅ PASSED")
                self.passed.append(name)
            else:
                print(f"❌ FAILED ({len(failed_checks)} checks)")
                self.failed.append(name)

                if self.verbose:
                    for check_name, error in failed_checks[:3]:  # Show first 3
                        error_msg = str(error).split('\n')[0][:100]
                        print(f"     └─ {check_name}: {error_msg}")
                    if len(failed_checks) > 3:
                        print(f"     └─ ... and {len(failed_checks) - 3} more")

            self.results[name] = {
                "passed": len(passed_checks),
                "failed": len(failed_checks),
                "errors": failed_checks,
            }

            if self.fail_fast and failed_checks:
                raise Exception(f"Stopping after first failure: {name}")

        except Exception as e:
            print(f"💥 CRASHED")
            self.failed.append(name)
            self.results[name] = {
                "passed": 0,
                "failed": -1,
                "error": str(e),
            }
            if self.verbose:
                print(f"     └─ {str(e)[:100]}")

    def print_summary(self):
        """Print test summary and statistics."""
        total = len(self.results)
        passed = len(self.passed)
        failed = len(self.failed)

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total detectors tested:  {total}")
        print(f"Passed all checks:       {passed:3d} ({100*passed//max(total,1):3d}%)")
        print(f"Failed checks:           {failed:3d} ({100*failed//max(total,1):3d}%)")
        print("=" * 60)

        if self.passed:
            print("\n✅ PASSED:")
            for name in sorted(self.passed):
                stats = self.results[name]
                print(f"   {name:35s} ({stats['passed']} checks)")

        if self.failed:
            print("\n❌ FAILED:")
            for name in sorted(self.failed):
                stats = self.results[name]
                failed_count = stats.get("failed", -1)
                if failed_count == -1:
                    print(f"   {name:35s} (crashed)")
                else:
                    passed_count = stats.get("passed", 0)
                    print(f"   {name:35s} ({passed_count} passed, {failed_count} failed)")

        print("\n" + "=" * 60)

    def print_detailed_failures(self):
        """Print detailed error messages for failed checks."""
        if not self.failed:
            return

        print("\n" + "=" * 60)
        print("DETAILED FAILURE ANALYSIS")
        print("=" * 60)

        for name in sorted(self.failed):
            stats = self.results[name]
            errors = stats.get("errors", [])

            if errors:
                print(f"\n{name}:")
                print("-" * 60)

                for check_name, error in errors[:5]:  # Show top 5 errors
                    print(f"\n  Check: {check_name}")
                    print(f"  Error: {type(error).__name__}")
                    error_lines = str(error).split('\n')
                    for line in error_lines[:3]:  # Show first 3 lines
                        print(f"    {line}")

                if len(errors) > 5:
                    print(f"\n  ... and {len(errors) - 5} more errors")

    def run(self):
        """Execute the full detector test suite."""
        print("\n" + "=" * 60)
        print("sktime ANOMALY DETECTOR API COMPLIANCE HUNTER")
        print("=" * 60)
        print("")

        detectors = self.get_detectors()
        if not detectors:
            print("No detectors found. Exiting.")
            return 1

        print("=" * 60)
        print("RUNNING API COMPLIANCE CHECKS")
        print("=" * 60)
        print("")

        for name, detector_cls in detectors:
            self.test_detector(name, detector_cls)

        self.print_summary()

        if self.verbose:
            self.print_detailed_failures()

        # Return exit code based on results
        return 0 if not self.failed else 1


def main():
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="sktime Anomaly Detector API Compliance Hunter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hunt.py                    # Test all detectors
  python hunt.py --verbose          # Detailed output
  python hunt.py --fail-fast        # Stop on first failure
  python hunt.py --exclude SubLOF   # Skip specific detector
        """,
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed error messages",
    )
    parser.add_argument(
        "-f", "--fail-fast",
        action="store_true",
        help="Stop after first detector failure",
    )
    parser.add_argument(
        "-e", "--exclude",
        nargs="+",
        default=[],
        help="Detector names to skip",
    )

    args = parser.parse_args()

    hunter = DetectorHunter(
        verbose=args.verbose,
        fail_fast=args.fail_fast,
        exclude=args.exclude,
    )

    exit_code = hunter.run()
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
