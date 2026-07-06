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

from sktime.benchmarking.post_hoc._base import BasePostHocEvaluator
from sktime.benchmarking.post_hoc._critical_difference import (
    CriticalDifferenceDiagram,
)
from sktime.benchmarking.post_hoc._friedman import FriedmanEvaluator
from sktime.benchmarking.post_hoc._nemenyi import NemenyiEvaluator
from sktime.benchmarking.post_hoc._rank import RankEvaluator
from sktime.benchmarking.post_hoc._ranksum import RanksumEvaluator
from sktime.benchmarking.post_hoc._sign import SignTestEvaluator
from sktime.benchmarking.post_hoc._ttest import TTestEvaluator
from sktime.benchmarking.post_hoc._wilcoxon import WilcoxonEvaluator
