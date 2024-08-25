"""Test orchestration."""

__author__ = ["viktorkaz", "mloning"]

import os

import numpy as np
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

# get data path for testing dataset loading from hard drive
import sktime
from sktime.benchmarking.data import RAMDataset, UEADataset
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import AggregateMetric, PairwiseMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults, RAMResults
from sktime.benchmarking.strategies import TSCStrategy
from sktime.benchmarking.tasks import TSCTask
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sktime.classification.ensemble import ComposableTimeSeriesForestClassifier
from sktime.datasets import load_arrow_head, load_gunpoint
from sktime.series_as_features.model_selection import SingleSplit
from sktime.tests.test_switch import run_test_module_changed
from sktime.transformations.panel.reduce import Tabularizer

REPOPATH = os.path.dirname(sktime.__file__)
DATAPATH = os.path.join(REPOPATH, "datasets/data/")


def make_reduction_pipeline(estimator):
    """Use tabular estimators in time series setting."""
    pipeline = Pipeline([("transform", Tabularizer()), ("clf", estimator)])
    return pipeline


# simple test of orchestration and metric evaluation
@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
@pytest.mark.parametrize("data_loader", [load_gunpoint, load_arrow_head])
def test_automated_orchestration_vs_manual(data_loader):
    """Test orchestration."""
    data = data_loader(return_X_y=False)

    dataset = RAMDataset(dataset=data, name="data")
    task = TSCTask(target="class_val")

    # create strategies
    # clf = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
    clf = make_reduction_pipeline(
        RandomForestClassifier(n_estimators=2, random_state=1)
    )
    strategy = TSCStrategy(clf)

    # result backend
    results = RAMResults()
    orchestrator = Orchestrator(
        datasets=[dataset],
        tasks=[task],
        strategies=[strategy],
        cv=SingleSplit(random_state=1),
        results=results,
    )

    orchestrator.fit_predict(save_fitted_strategies=False)
    result = next(results.load_predictions(cv_fold=0, train_or_test="test"))  # get
    # only first item of iterator
    actual = result.y_pred

    # expected output
    task = TSCTask(target="class_val")
    cv = SingleSplit(random_state=1)
    train_idx, test_idx = next(cv.split(data))
    train = data.iloc[train_idx, :]
    test = data.iloc[test_idx, :]
    strategy.fit(task, train)
    expected = strategy.predict(test)

    # compare results
    np.testing.assert_array_equal(actual, expected)


# extensive tests of orchestration and metric evaluation against sklearn
@pytest.mark.skip(reason="failures since sklearn 1.4.0, see #5797")
@pytest.mark.parametrize(
    "dataset",
    [
        RAMDataset(dataset=load_arrow_head(return_X_y=False), name="ArrowHead"),
        UEADataset(path=DATAPATH, name="GunPoint", target_name="class_val"),
    ],
)
@pytest.mark.parametrize(
    "cv", [SingleSplit(random_state=1), StratifiedKFold(random_state=1, shuffle=True)]
)
@pytest.mark.parametrize(
    "metric_func",
    [accuracy_score, f1_score],  # pairwise metric  # composite metric
)
@pytest.mark.parametrize("results_cls", [RAMResults, HDDResults])
@pytest.mark.parametrize(
    "estimator",
    [
        DummyClassifier(strategy="most_frequent", random_state=1),
        RandomForestClassifier(n_estimators=2, random_state=1),
    ],
)
def test_single_dataset_single_strategy_against_sklearn(
    dataset, cv, metric_func, estimator, results_cls, tmpdir
):
    """Test against sklearn."""
    # set up orchestration
    task = TSCTask(target="class_val")

    # create strategies
    clf = make_reduction_pipeline(estimator)
    strategy = TSCStrategy(clf)

    # result backend
    if results_cls in [HDDResults]:
        # for hard drive results, create temporary directory using pytest's
        # tmpdir fixture
        tempdir = tmpdir.mkdir("results/")
        path = tempdir.dirpath()
        results = results_cls(path=path)
    elif results_cls in [RAMResults]:
        results = results_cls()
    else:
        raise ValueError()

    orchestrator = Orchestrator(
        datasets=[dataset], tasks=[task], strategies=[strategy], cv=cv, results=results
    )
    orchestrator.fit_predict(save_fitted_strategies=False)

    evaluator = Evaluator(results)

    # create metric classes for evaluation and set metric kwargs
    if metric_func in [accuracy_score]:
        kwargs = {}  # empty kwargs for simple pairwise metrics
        metric = PairwiseMetric(func=metric_func, name="metric")
    elif metric_func in [f1_score]:
        kwargs = {"average": "macro"}  # set kwargs for composite metrics
        metric = AggregateMetric(func=metric_func, name="metric", **kwargs)
    else:
        raise ValueError()

    metrics = evaluator.evaluate(metric=metric)
    actual = metrics["metric_mean"].iloc[0]

    # compare against sklearn cross_val_score
    data = dataset.load()  # load data
    X = data.loc[:, task.features]
    y = data.loc[:, task.target]
    expected = cross_val_score(
        clf, X, y, scoring=make_scorer(metric_func, **kwargs), cv=cv
    ).mean()

    # compare results
    np.testing.assert_array_equal(actual, expected)


# simple test of sign test and ranks
@pytest.mark.skipif(
    not run_test_module_changed("sktime.benchmarking"),
    reason="run test only if benchmarking module has changed",
)
def test_stat():
    """Test sign ranks."""
    data = load_gunpoint(split="train", return_X_y=False)[:7]
    dataset = RAMDataset(dataset=data, name="gunpoint")
    task = TSCTask(target="class_val")

    fc = ComposableTimeSeriesForestClassifier(n_estimators=1, random_state=1)
    strategy_fc = TSCStrategy(fc, name="tsf")
    pf = KNeighborsTimeSeriesClassifier()
    strategy_pf = TSCStrategy(pf, name="pf")

    # result backend
    results = RAMResults()
    orchestrator = Orchestrator(
        datasets=[dataset],
        tasks=[task],
        strategies=[strategy_pf, strategy_fc],
        cv=SingleSplit(random_state=1),
        results=results,
    )

    orchestrator.fit_predict(save_fitted_strategies=False)

    analyse = Evaluator(results)
    metric = PairwiseMetric(func=accuracy_score, name="accuracy")
    _ = analyse.evaluate(metric=metric)

    ranks = analyse.rank(ascending=True)
    pf_rank = ranks.loc[ranks.strategy == "pf", "accuracy_mean_rank"].item()  # 1
    fc_rank = ranks.loc[ranks.strategy == "tsf", "accuracy_mean_rank"].item()  # 2
    rank_array = [pf_rank, fc_rank]
    rank_array_test = [2, 1]
    _, sign_test_df = analyse.sign_test()

    sign_array = [
        [sign_test_df["pf"].iloc[0], sign_test_df["pf"].iloc[1]],
        [sign_test_df["tsf"].iloc[0], sign_test_df["tsf"].iloc[1]],
    ]
    sign_array_test = [[1, 1], [1, 1]]
    np.testing.assert_equal(
        [rank_array, sign_array], [rank_array_test, sign_array_test]
    )
