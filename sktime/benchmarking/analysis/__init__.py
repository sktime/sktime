"""Analyzers for benchmark results (v2, strategy pattern).

These evaluators consume the flat results table produced by the v2 benchmarking
framework (``BaseBenchmark.run`` / ``ResultObject.to_dataframe``) and compute
analysis results: ranking, omnibus / pairwise significance tests, and the
critical-difference diagram.
"""

__all__ = [
    "BaseBenchmarkAnalyzer",
    "RankEvaluator",
    "FriedmanEvaluator",
    "NemenyiEvaluator",
    "WilcoxonEvaluator",
    "SignTestEvaluator",
    "RanksumEvaluator",
    "TTestEvaluator",
    "CriticalDifferenceDiagram",
]

from sktime.benchmarking.analysis._base import BaseBenchmarkAnalyzer
from sktime.benchmarking.analysis._critical_difference import (
    CriticalDifferenceDiagram,
)
from sktime.benchmarking.analysis._friedman import FriedmanEvaluator
from sktime.benchmarking.analysis._nemenyi import NemenyiEvaluator
from sktime.benchmarking.analysis._rank import RankEvaluator
from sktime.benchmarking.analysis._ranksum import RanksumEvaluator
from sktime.benchmarking.analysis._sign import SignTestEvaluator
from sktime.benchmarking.analysis._ttest import TTestEvaluator
from sktime.benchmarking.analysis._wilcoxon import WilcoxonEvaluator
