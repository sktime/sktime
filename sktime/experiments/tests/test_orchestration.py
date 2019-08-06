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
