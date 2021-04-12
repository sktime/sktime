# -*- coding: utf-8 -*-
__author__ = ["Viktor Kazakov", "Markus Löning"]

import os

import numpy as np
import pytest

# get data path for testing dataset loading from hard drive
import sktime
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sktime.benchmarking.data import RAMDataset
from sktime.benchmarking.data import UEADataset
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import AggregateMetric
from sktime.benchmarking.metrics import PairwiseMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import HDDResults
from sktime.benchmarking.results import RAMResults
from sktime.benchmarking.strategies import TSCStrategy
from sktime.benchmarking.tasks import TSCTask
from sktime.classification.distance_based._proximity_forest import ProximityForest
from sktime.datasets import load_arrow_head
from sktime.datasets import load_gunpoint
from sktime.classification.compose import ComposableTimeSeriesForestClassifier
from sktime.series_as_features.model_selection import SingleSplit
from sktime.transformations.panel.reduce import Tabularizer

REPOPATH = os.path.dirname(sktime.__file__)
DATAPATH = os.path.join(REPOPATH, "datasets/data/")


def make_reduction_pipeline(estimator):
    """Helper function to use tabular estimators in time series setting"""
    pipeline = Pipeline([("transform", Tabularizer()), ("clf", estimator)])
    return pipeline


# simple test of orchestration and metric evaluation
@pytest.mark.parametrize("data_loader", [load_gunpoint, load_arrow_head])
def test_automated_orchestration_vs_manual(data_loader):
    data = data_loader()

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
@pytest.mark.parametrize(
    "dataset",
    [
        RAMDataset(dataset=load_arrow_head(), name="ArrowHead"),
        UEADataset(path=DATAPATH, name="GunPoint", target_name="class_val"),
    ],
)
@pytest.mark.parametrize(
    "cv", [SingleSplit(random_state=1), StratifiedKFold(random_state=1, shuffle=True)]
)
@pytest.mark.parametrize(
    "metric_func", [accuracy_score, f1_score]  # pairwise metric  # composite metric
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
def test_stat():
    data = load_gunpoint(split="train")
    dataset = RAMDataset(dataset=data, name="gunpoint")
    task = TSCTask(target="class_val")

    fc = ComposableTimeSeriesForestClassifier(n_estimators=1, random_state=1)
    strategy_fc = TSCStrategy(fc, name="tsf")
    pf = ProximityForest(n_estimators=1, random_state=1)
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
    rank_array_test = [1, 2]
    _, sign_test_df = analyse.sign_test()

    sign_array = [
        [sign_test_df["pf"][0], sign_test_df["pf"][1]],
        [sign_test_df["tsf"][0], sign_test_df["tsf"][1]],
    ]
    sign_array_test = [[1, 1], [1, 1]]
    np.testing.assert_equal(
        [rank_array, sign_array], [rank_array_test, sign_array_test]
    )
