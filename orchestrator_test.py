from sktime.experiments.orchestrator import Orchestrator
from sktime.model_selection import Single_Split
import sktime
from sktime.highlevel import Task, TSCStrategy
from sktime.datasets import load_gunpoint
from sktime.classifiers.ensemble import TimeSeriesForestClassifier
import pandas as pd

from sktime.experiments.data import DataHolder

from sktime.analyze_results import AnalyseResults
from sktime.analyze_results.scores import ScoreAccuracy
from sktime.experiments.data import DataLoader

#from memory example
train = load_gunpoint(split='TRAIN')
test = load_gunpoint(split='TEST')
data = pd.concat([train,test], axis=0)
task = Task(case='TSC', data=data, dataset_name='gunpoint',target='label')
dh = DataHolder(data=data, task=task, dataset_name='GunPoint')

clf = TimeSeriesForestClassifier()
strategy = TSCStrategy(clf)

orchestrator = Orchestrator(save_results=True)


# results = orchestrator.run_from_memory(data_holders=[dh], 
#                                         strategies=[strategy], 
#                                         resampling=Single_Split())

#from disk example
dl = DataLoader(dts_dir='data/datasets', task_types='TSC')
results = orchestrator.fit(data=dl,
                            strategies=[strategy])


# results = orchestrator.fit(data=[dh],
#                             strategies=[strategy],
#                             resampling=Single_Split())

analyze = AnalyseResults(results)
losses, losses_df = analyze.prediction_errors(metric=ScoreAccuracy())

avg_and_std = analyze.average_and_std_error(losses)


print(avg_and_std)

