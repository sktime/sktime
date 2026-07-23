"""Tests that the deprecated v1 benchmarking framework emits FutureWarning."""

import contextlib
import warnings

import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

from sktime.benchmarking.base import BaseDataset, BaseResults
from sktime.benchmarking.data import RAMDataset, UEADataset, make_datasets
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.experiments import (
    load_and_run_classification_experiment,
    load_and_run_clustering_experiment,
    run_classification_experiment,
    run_clustering_experiment,
)
from sktime.benchmarking.metrics import AggregateMetric, PairwiseMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults, RAMResults
from sktime.benchmarking.strategies import TSCStrategy, TSRStrategy
from sktime.benchmarking.tasks import BaseTask, TSCTask, TSRTask
from sktime.tests.test_switch import run_test_module_changed

# substring of the deprecation message, see benchmarking.base._V1_DEPRECATION_MSG
_MATCH = "v1 benchmarking framework is deprecated"


def _evaluator():
    # build the (also deprecated) results object silently so the assertion below
    # only captures the Evaluator deprecation warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        results = RAMResults()
    return Evaluator(results)


# one entry per deprecated v1 public entry point; the value constructs / calls it
V1_ENTRY_POINTS = {
    "BaseResults": lambda: BaseResults(),
    "RAMResults": lambda: RAMResults(),
    "HDDResults": lambda: HDDResults(path="nonexistent_v1_results"),
    "BaseDataset": lambda: BaseDataset("d"),
    "RAMDataset": lambda: RAMDataset(pd.DataFrame({"a": [1]}), "d"),
    "UEADataset": lambda: UEADataset(path="x", name="y"),
    "make_datasets": lambda: make_datasets("x", UEADataset, names=[]),
    "BaseTask": lambda: BaseTask(target="t"),
    "TSCTask": lambda: TSCTask(target="t"),
    "TSRTask": lambda: TSRTask(target="t"),
    "TSCStrategy": lambda: TSCStrategy(object()),
    "TSRStrategy": lambda: TSRStrategy(object()),
    "PairwiseMetric": lambda: PairwiseMetric(func=accuracy_score, name="acc"),
    "AggregateMetric": lambda: AggregateMetric(func=accuracy_score, name="acc"),
    "Orchestrator": lambda: Orchestrator(None, None, None, None, None),
    "Evaluator": _evaluator,
    "run_clustering_experiment": lambda: run_clustering_experiment(None, None, None),
    "load_and_run_clustering_experiment": (
        lambda: load_and_run_clustering_experiment(None, None, None, None)
    ),
    "run_classification_experiment": (
        lambda: run_classification_experiment(None, None, None, None, None, None)
    ),
    "load_and_run_classification_experiment": (
        lambda: load_and_run_classification_experiment(None, None, None, None)
    ),
}


@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("name", list(V1_ENTRY_POINTS), ids=list(V1_ENTRY_POINTS))
def test_v1_entrypoint_warns(name):
    """Each deprecated v1 entry point emits a FutureWarning on use.

    The call is wrapped in ``suppress`` because the deprecation warning is raised
    as the first statement, before any (here intentionally invalid) arguments are
    processed - so the warning fires even when the subsequent call fails.
    """
    with pytest.warns(FutureWarning, match=_MATCH):
        with contextlib.suppress(Exception):
            V1_ENTRY_POINTS[name]()
