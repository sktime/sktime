from sktime.experiments.orchestrator import Orchestrator
from sktime.highlevel import TSCTask, TSCStrategy
from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sktime.experiments.data import DatasetRAM
from sktime.experiments.data import ResultRAM
from sktime.model_selection import SingleSplit
from sklearn.dummy import DummyClassifier
from sktime.experiments.analysis import AnalyseResults
from sktime.experiments.scores import ScoreAccuracy
import pandas as pd
from sktime.datasets import load_gunpoint

def test_orchestration():
    X_train, y_train = load_gunpoint(return_X_y=True)
    data=pd.concat([X_train, y_train], axis=1)
    data.columns=['dim_0','target']

    dataset = DatasetRAM(dataset=data, dataset_name='gunpoint')
    task = TSCTask(target='target')

    #create strategies
    clf = DummyClassifier(random_state=1)
    strategy = TSCStrategy(clf)


    #result backend
    resultRAM = ResultRAM()
    orchestrator = Orchestrator(datasets=[dataset],
                                tasks=[task],  
                                strategies=[strategy], 
                                cv=SingleSplit(random_state=1), 
                                result=resultRAM)

    predictions = ['1', '2', '1', '1', '1', '1', '1', '1', '1', '2', '1', '2', '1', '2', '1', '2', '1', '2', '1', '1', '2', '2', '1', '2', '2', '2', '1', '1', '1', '2', '1', '1', '2', '2', '2', '1', '2', '2', '1', '2', '2', '2', '1', '2', '1', '1', '2', '1', '1', '1']



    orchestrator.run(save_strategies=False)
    result = resultRAM.load()
    assert result[0].y_pred == predictions
