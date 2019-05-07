from sktime.experiments.orchestrator import Orchestrator
from sktime.highlevel import TSCTask, TSCStrategy
from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sktime.model_selection import PresplitFilesCV
from sktime.experiments.data import ResultHDD
from sktime.experiments.data import DatasetLoadFromDir
from sktime import datasets as data_module

from tempfile import TemporaryDirectory
import os

TEMPDIR = "temp/"
DATADIR = os.path.join(os.path.dirname(data_module.__file__), 'data')


def test_orchestration():
    data = DatasetLoadFromDir(root_dir=DATADIR)
    datasets = data.load_datasets()
    n_datasets = len(datasets)
    tasks = [TSCTask(target='target') for _ in range(n_datasets)]

    estimator = TimeSeriesForestClassifier(n_estimators=2)
    strategies = [TSCStrategy(estimator) for _ in range(2)]

    with TemporaryDirectory(TEMPDIR):
        results = ResultHDD(results_save_dir=TEMPDIR)

        orchestrator = Orchestrator(datasets=datasets,
                                    tasks=tasks,
                                    strategies=strategies,
                                    cv=PresplitFilesCV(),
                                    result=results)
        orchestrator.run()
