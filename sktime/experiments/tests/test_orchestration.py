import numpy as np

from sktime.datasets import load_gunpoint
from sktime.experiments.data import DatasetRAM
from sktime.experiments.data import ResultRAM
from sktime.experiments.orchestrator import Orchestrator
from sktime.highlevel.tasks import TSCTask
from sktime.highlevel.strategies import TSCStrategy
from sktime.model_selection import SingleSplit
from sktime.classifiers.compose.ensemble import TimeSeriesForestClassifier


def test_orchestration():
    data = load_gunpoint()

    dataset = DatasetRAM(dataset=data, dataset_name='gunpoint')
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
                                result=resultRAM)

    orchestrator.run(save_strategies=False)
    result = resultRAM.load()
    actual = result[0].y_pred

    # expected output
    task = TSCTask(target='class_val')
    cv = SingleSplit(random_state=1)
    for train_idx, test_idx in cv.split(data):
        train = data.iloc[train_idx, :]
        test = data.iloc[test_idx, :]
        clf = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
        strategy = TSCStrategy(clf)
        strategy.fit(task, train)
        expected = strategy.predict(test)

    np.testing.assert_array_equal(actual, expected)

def test_accuracy():
    data = load_gunpoint()

    dataset = DatasetRAM(dataset=data, dataset_name='gunpoint')
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
                                result=resultRAM)

    orchestrator.run(save_strategies=False)
    
    analyse = AnalyseResults(resultRAM)
    strategy_dict, losses_df = analyse.prediction_errors(metric= ScoreAccuracy())
    
    train_accuracy = strategy_dict['TimeSeriesForestClassifier'][0]
    test_accuracy = 0.8461538461538461

    np.testing.assert_equal(train_accuracy, test_accuracy)


def test_stat():
    data = load_gunpoint()
    dataset = DatasetRAM(dataset=data, dataset_name='gunpoint')
    task = TSCTask(target='class_val')
    
    fc = TimeSeriesForestClassifier(n_estimators=1, random_state=1)
    strategy_fc = TSCStrategy(fc)
    pf = ProximityForest(num_trees=1, num_stump_evaluations=1, random_state=1)
    strategy_pf = TSCStrategy(pf)

    # result backend
    resultRAM = ResultRAM()
    orchestrator = Orchestrator(datasets=[dataset],
                                tasks=[task],
                                strategies=[strategy_pf, strategy_fc],
                                cv=SingleSplit(random_state=1),
                                result=resultRAM)

    orchestrator.run(save_strategies=False)

    analyse = AnalyseResults(resultRAM)
    strategy_dict, losses_df = analyse.prediction_errors(metric= ScoreAccuracy())

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


