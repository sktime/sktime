"""Main configuration file for pytest.

Contents:
adds the following options to pytest
--matrixdesign : bool, default False
    allows to turn on/off the sub-sampling in the tests (for shorter runtime)
    "on" condition is partition/block design to ensure each estimator full tests are run
    on each operating system at least once, and on each python version at least once,
    but not necessarily on each operating system / python version combination
--only_changed_modules : bool, default False
    turns on/off differential testing (for shorter runtime)
    "on" condition ensures that only estimators are tested that have changed,
    more precisely, only estimators whose class is in a module
    that has changed compared to the main branch
    "off" = runs tests for all estimators
--memory_profile : bool, default False
    turns on/off memory profiling of individual tests via tracemalloc
    "on" tracks peak memory per test and writes a report to memory_profile_report.txt
    useful for diagnosing memory-related CI/CD failures

by default, all options are off, including for default local runs of pytest
if multiple options are turned on, they are combined with AND,
i.e., intersection of estimators satisfying the conditions
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]

import tracemalloc

_memory_records = []


def pytest_addoption(parser):
    """Pytest command line parser options adder."""
    parser.addoption(
        "--matrixdesign",
        default=False,
        help="sub-sample estimators in tests by os/version matrix partition design",
    )
    parser.addoption(
        "--only_changed_modules",
        default=False,
        help="test only estimators from modules that have changed compared to main",
    )
    parser.addoption(
        "--memory_profile",
        default=False,
        help="track peak memory per test via tracemalloc, write memory_profile_report.txt",
    )


def pytest_configure(config):
    """Pytest configuration preamble."""
    from sktime.tests import _config

    if config.getoption("--matrixdesign") in [True, "True"]:
        _config.MATRIXDESIGN = True
    if config.getoption("--only_changed_modules") in [True, "True"]:
        _config.ONLY_CHANGED_MODULES = True


def _memory_profiling_on(config):
    return config.getoption("--memory_profile") in [True, "True"]


def pytest_runtest_setup(item):
    """Start tracemalloc before each test if memory profiling is enabled."""
    if _memory_profiling_on(item.config):
        tracemalloc.start()


def pytest_runtest_teardown(item, nextitem):
    """Record peak memory after each test and stop tracemalloc."""
    if not _memory_profiling_on(item.config) or not tracemalloc.is_tracing():
        return

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    _memory_records.append((item.nodeid, peak))


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print top memory consumers and write full report to file."""
    if not _memory_profiling_on(config) or not _memory_records:
        return

    sorted_records = sorted(_memory_records, key=lambda x: x[1], reverse=True)

    report_path = "memory_profile_report.txt"
    with open(report_path, "w") as fh:
        fh.write("sktime pytest memory profile report\n")
        fh.write("=" * 72 + "\n")
        fh.write(f"{'Peak memory (MiB)':>20}  Test\n")
        fh.write("-" * 72 + "\n")
        for nodeid, peak_bytes in sorted_records:
            fh.write(f"{peak_bytes / (1024 ** 2):>20.3f}  {nodeid}\n")

    terminalreporter.write_sep("=", "top 20 tests by peak memory (tracemalloc)")
    for nodeid, peak_bytes in sorted_records[:20]:
        terminalreporter.write_line(f"{peak_bytes / (1024 ** 2):>10.3f} MiB  {nodeid}")
    terminalreporter.write_line(
        f"\nFull report written to: {report_path} ({len(sorted_records)} tests)"
    )
