from sktime.experiments.orchestrator import Orchestrator
from sktime.model_selection import Single_Split
import sktime
from sktime.highlevel import Task, TSCStrategy
from sktime.datasets import load_gunpoint
from sktime.classifiers.ensemble import TimeSeriesForestClassifier
import pandas as pd

from sktime.highlevel import DataHolder

from sktime.analyze_results import AnalyseResults
from sktime.analyze_results.scores import ScoreAccuracy

train = load_gunpoint(split='TRAIN')
test = load_gunpoint(split='TEST')
data = pd.concat([train,test], axis=0)
task = Task(case='TSC', data=data, dataset_name='gunpoint',target='label')
dh = DataHolder(data=data, task=task, dataset_name='GunPoint')

clf = TimeSeriesForestClassifier()
strategy = TSCStrategy(clf)

orchestrator = Orchestrator(data_holders=[dh], strategies=[strategy], resampling=Single_Split())


results = orchestrator.run()

analyze = AnalyseResults(results)
losses = analyze.prediction_errors(metric=ScoreAccuracy())

avg_and_std = analyze.average_and_std_error(losses)


print(avg_and_std)

