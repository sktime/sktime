import numpy as np

from sktime.benchmarking.data import RAMDataset
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import Accuracy
from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.results import RAMResults
from sktime.classifiers.compose.ensemble import TimeSeriesForestClassifier
from sktime.classifiers.distance_based.proximity_forest import ProximityForest
from sktime.datasets import load_gunpoint
from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask
from sktime.model_selection import SingleSplit


def test_orchestration():
    data = load_gunpoint()

    dataset = RAMDataset(dataset=data, name='gunpoint')
    task = TSCTask(target='class_val')

    # create strategies
    clf = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
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
    actual = np.array(result.y_pred, dtype=np.intp)

    # expected output
    task = TSCTask(target='class_val')
    cv = SingleSplit(random_state=1)
    for train_idx, test_idx in cv.split(data):
        train = data.iloc[train_idx, :]
        test = data.iloc[test_idx, :]
        clf = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
        strategy = TSCStrategy(clf)
        strategy.fit(task, train)
        expected = np.array(strategy.predict(test), dtype=np.intp)

    np.testing.assert_array_equal(actual, expected)


def test_accuracy():
    data = load_gunpoint()

    dataset = RAMDataset(dataset=data, name='gunpoint')
    task = TSCTask(target='class_val')

    # create strategies
    clf = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
    strategy = TSCStrategy(clf)

    # result backend
    results = RAMResults()
    orchestrator = Orchestrator(datasets=[dataset],
                                tasks=[task],
                                strategies=[strategy],
                                cv=SingleSplit(random_state=1),
                                results=results)

    orchestrator.fit_predict(save_fitted_strategies=False)

    analyse = Evaluator(results)
    losses_df = analyse.evaluate(metric=Accuracy())

    testing_loss = losses_df['mean'].iloc[0]
    true_loss = 1 - 0.15384615384615385

    np.testing.assert_equal(true_loss, testing_loss)


def test_stat():
    data = load_gunpoint()
    dataset = RAMDataset(dataset=data, name='gunpoint')
    task = TSCTask(target='class_val')

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
    losses_df = analyse.evaluate(metric=Accuracy())

    ranks = analyse.ranks()
    pf_rank = ranks.loc['ProximityForest'][0]  # 1
    fc_rank = ranks.loc['TimeSeriesForestClassifier'][0]  # 2
    rank_array = [pf_rank, fc_rank]
    rank_array_test = [1, 2]
    _, sign_test_df = analyse.sign_test()

    sign_array = [
        [sign_test_df['ProximityForest'][0], sign_test_df['ProximityForest'][1]],
        [sign_test_df['TimeSeriesForestClassifier'][0], sign_test_df['TimeSeriesForestClassifier'][1]]
    ]
    sign_array_test = [[1, 1], [1, 1]]
    np.testing.assert_equal([rank_array, sign_array], [rank_array_test, sign_array_test])
