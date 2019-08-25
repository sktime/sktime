__author__ = ["Viktor Kazakov", "Markus Löning"]

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, zero_one_loss, average_precision_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

from sktime.benchmarking.data import RAMDataset
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import PointWiseMetric, CompositeMetric
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import RAMResults
from sktime.classifiers.compose.ensemble import TimeSeriesForestClassifier
from sktime.classifiers.distance_based.proximity_forest import ProximityForest
from sktime.datasets import load_gunpoint, load_arrow_head
from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask
from sktime.model_selection import SingleSplit
from sktime.pipeline import Pipeline
from sktime.transformers.compose import Tabulariser


# specify strategies
def make_reduction_pipeline(estimator):
    pipeline = Pipeline([
        ("transform", Tabulariser()),
        ("clf", estimator)
    ])
    return pipeline


@pytest.mark.parametrize("data_loader", [load_gunpoint, load_arrow_head])
def test_orchestration(data_loader):
    data = data_loader()

    dataset = RAMDataset(dataset=data, name="data")
    task = TSCTask(target="class_val")

    # create strategies
    # clf = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
    clf = make_reduction_pipeline(RandomForestClassifier(n_estimators=1, random_state=1))
    strategy = TSCStrategy(clf)

    # result backend
    results = RAMResults()
    orchestrator = Orchestrator(datasets=[dataset],
                                tasks=[task],
                                strategies=[strategy],
                                cv=SingleSplit(random_state=1),
                                results=results)

    orchestrator.fit_predict(save_fitted_strategies=False)
    result = next(results.load_predictions())  # get only first item of iterator
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


@pytest.mark.parametrize("data_loader", [load_gunpoint, load_arrow_head])
@pytest.mark.parametrize("cv", [SingleSplit, KFold])
@pytest.mark.parametrize("metric", [accuracy_score, zero_one_loss])
def test_against_sklearn(data_loader, cv, metric):
    data = data_loader()
    cv = cv(random_state=1)

    # setup orchestration
    dataset = RAMDataset(dataset=data, name="data")
    task = TSCTask(target="class_val")

    # create strategies
    clf = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
    strategy = TSCStrategy(clf)

    # result backend
    results = RAMResults()
    orchestrator = Orchestrator(datasets=[dataset],
                                tasks=[task],
                                strategies=[strategy],
                                cv=cv,
                                results=results)
    orchestrator.fit_predict(save_fitted_strategies=False)

    analyse = Evaluator(results)

    # create metric for evaluation
    if metric in [accuracy_score, zero_one_loss]:
        metric_cls = PointWiseMetric(func=metric, name="metric")
    elif metric in [average_precision_score]:
        metric_cls = CompositeMetric(func=metric, name="metric")
    else:
        raise ValueError()
    metrics = analyse.evaluate(metric=metric_cls)
    actual = metrics["metric_mean"].iloc[0]

    # compare against sklearn cross_val_score
    X = data.loc[:, task.features]
    y = data.loc[:, task.target]
    expected = np.mean(cross_val_score(clf, X, y,
                                       scoring=make_scorer(metric),
                                       cv=cv))

    np.testing.assert_equal(actual, expected)


def test_stat():
    data = load_gunpoint()
    dataset = RAMDataset(dataset=data, name="gunpoint")
    task = TSCTask(target="class_val")

    fc = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
    strategy_fc = TSCStrategy(fc)
    pf = ProximityForest(n_trees=1, random_state=1)
    strategy_pf = TSCStrategy(pf)

    # result backend
    results = RAMResults()
    orchestrator = Orchestrator(datasets=[dataset],
                                tasks=[task],
                                strategies=[strategy_pf, strategy_fc],
                                cv=SingleSplit(random_state=1),
                                results=results)

    orchestrator.fit_predict(save_fitted_strategies=False)

    analyse = Evaluator(results)
    metric = PointWiseMetric(func=accuracy_score, name="accuracy")
    losses_df = analyse.evaluate(metric=metric)

    ranks = analyse.rank(ascending=True)
    pf_rank = ranks.loc["ProximityForest"][0]  # 1
    fc_rank = ranks.loc["TimeSeriesForestClassifier"][0]  # 2
    rank_array = [pf_rank, fc_rank]
    rank_array_test = [1, 2]
    _, sign_test_df = analyse.sign_test()

    sign_array = [
        [sign_test_df["ProximityForest"][0], sign_test_df["ProximityForest"][1]],
        [sign_test_df["TimeSeriesForestClassifier"][0], sign_test_df["TimeSeriesForestClassifier"][1]]
    ]
    sign_array_test = [[1, 1], [1, 1]]
    np.testing.assert_equal([rank_array, sign_array], [rank_array_test, sign_array_test])
