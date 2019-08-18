import numpy as np

from sktime.datasets import load_gunpoint
from sktime.benchmarking.data import DatasetRAM
from sktime.benchmarking.data import ResultRAM
from sktime.benchmarking.orchestration import Orchestrator
from sktime.highlevel.tasks import TSCTask
from sktime.highlevel.strategies import TSCStrategy
from sktime.model_selection import SingleSplit
from sktime.classifiers.compose.ensemble import TimeSeriesForestClassifier
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import ScoreAccuracy
from sktime.classifiers.distance_based.proximity_forest import ProximityForest

def test_orchestration():
    data = load_gunpoint()

    dataset = DatasetRAM(dataset=data, name='gunpoint')
    task = TSCTask(target='class_val')

    # create strategies
    clf = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
    strategy = TSCStrategy(clf)

    # result backend
    resultRAM = ResultRAM()
    orchestrator = Orchestrator(datasets=[dataset],
                                tasks=[task],
                                strategies=[strategy],
                                cv=SingleSplit(random_state=1),
                                results=resultRAM)

    orchestrator.fit_predict(save_fitted_strategies=False)
    result = resultRAM.load_predictions()
    actual = np.array(result[0].y_pred, dtype=np.intp)

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

    dataset = DatasetRAM(dataset=data, name='gunpoint')
    task = TSCTask(target='class_val')

    # create strategies
    clf = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
    strategy = TSCStrategy(clf)

    # result backend
    resultRAM = ResultRAM()
    orchestrator = Orchestrator(datasets=[dataset],
                                tasks=[task],
                                strategies=[strategy],
                                cv=SingleSplit(random_state=1),
                                results=resultRAM)

    orchestrator.fit_predict(save_fitted_strategies=False)
    
    analyse = Evaluator(resultRAM)
    strategy_dict, losses_df = analyse.compute_metric(metric= ScoreAccuracy())
    
    testing_loss = losses_df['loss'].iloc[0]
    true_loss = 0.15384615384615385

    np.testing.assert_equal(true_loss, testing_loss)


def test_stat():
    data = load_gunpoint()
    dataset = DatasetRAM(dataset=data, name='gunpoint')
    task = TSCTask(target='class_val')
    
    fc = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
    strategy_fc = TSCStrategy(fc)
    pf = ProximityForest(n_trees=1, random_state=1)
    strategy_pf = TSCStrategy(pf)

    # result backend
    resultRAM = ResultRAM()
    orchestrator = Orchestrator(datasets=[dataset],
                                tasks=[task],
                                strategies=[strategy_pf, strategy_fc],
                                cv=SingleSplit(random_state=1),
                                results=resultRAM)

    orchestrator.fit_predict(save_fitted_strategies=False)

    analyse = Evaluator(resultRAM)
    strategy_dict, losses_df = analyse.compute_metric(metric= ScoreAccuracy())

    ranks = analyse.ranks(strategy_dict)
    pf_rank = ranks.loc['ProximityForest'][0] #1
    fc_rank = ranks.loc['TimeSeriesForestClassifier'][0] #2
    rank_array = [pf_rank, fc_rank]
    rank_array_test = [1,2]
    sign_test, sign_test_df = analyse.sign_test(strategy_dict)
    
    sign_array = [
        [sign_test_df['ProximityForest'][0], sign_test_df['ProximityForest'][1]],
        [sign_test_df['TimeSeriesForestClassifier'][0], sign_test_df['TimeSeriesForestClassifier'][1]]
    ]
    sign_array_test = [[1,1],[1,1]]
    np.testing.assert_equal([rank_array, sign_array],[rank_array_test,sign_array_test])

