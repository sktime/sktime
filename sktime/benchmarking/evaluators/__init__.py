"""Post-hoc statistical evaluators for benchmark results (v2, strategy pattern).

These evaluators consume the flat results table produced by the v2 benchmarking
framework (``BaseBenchmark.run`` / ``ResultObject.to_dataframe``) and compute
post-hoc statistical analyses that were previously locked inside the legacy v1
``Evaluator``: ranking, omnibus / pairwise significance tests, and the
critical-difference diagram.
"""

__all__ = [
    "BasePostHocEvaluator",
    "RankEvaluator",
    "FriedmanEvaluator",
    "NemenyiEvaluator",
    "WilcoxonEvaluator",
    "SignTestEvaluator",
    "RanksumEvaluator",
    "TTestEvaluator",
    "CriticalDifferenceDiagram",
]

from sktime.benchmarking.evaluators._base import BasePostHocEvaluator
from sktime.benchmarking.evaluators._critical_difference import (
    CriticalDifferenceDiagram,
)
from sktime.benchmarking.evaluators._friedman import FriedmanEvaluator
from sktime.benchmarking.evaluators._nemenyi import NemenyiEvaluator
from sktime.benchmarking.evaluators._rank import RankEvaluator
from sktime.benchmarking.evaluators._ranksum import RanksumEvaluator
from sktime.benchmarking.evaluators._sign import SignTestEvaluator
from sktime.benchmarking.evaluators._ttest import TTestEvaluator
from sktime.benchmarking.evaluators._wilcoxon import WilcoxonEvaluator
