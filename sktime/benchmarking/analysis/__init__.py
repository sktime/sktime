"""Analyzers for benchmark results (v2, strategy pattern).

These evaluators consume the flat results table produced by the v2 benchmarking
framework (``BaseBenchmark.run`` / ``ResultObject.to_dataframe``) and compute
analysis results: ranking, omnibus / pairwise significance tests, and the
critical-difference diagram.
"""

__all__ = [
    "BaseBenchmarkAnalyzer",
    "AverageRank",
    "FriedmanTest",
    "NemenyiTest",
    "WilcoxonSignedRankTest",
    "SignTest",
    "RankSumTest",
    "TwoSampleTTest",
    "CriticalDifferenceDiagram",
]

from sktime.benchmarking.analysis._base import BaseBenchmarkAnalyzer
from sktime.benchmarking.analysis._critical_difference import (
    CriticalDifferenceDiagram,
)
from sktime.benchmarking.analysis._friedman import FriedmanTest
from sktime.benchmarking.analysis._nemenyi import NemenyiTest
from sktime.benchmarking.analysis._rank import AverageRank
from sktime.benchmarking.analysis._ranksum import RankSumTest
from sktime.benchmarking.analysis._sign import SignTest
from sktime.benchmarking.analysis._ttest import TwoSampleTTest
from sktime.benchmarking.analysis._wilcoxon import WilcoxonSignedRankTest
