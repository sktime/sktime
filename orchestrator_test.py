from sktime.experiments.orchestrator import Orchestrator
from sktime.resampling.single_split import Single_Split
import sktime
from sktime.highlevel import Task, TSCStrategy
from sktime.datasets import load_gunpoint
from sktime.classifiers.ensemble import TimeSeriesForestClassifier

train = load_gunpoint(split='TRAIN')
test = load_gunpoint(split='TEST')

task = Task(case='TSC', data=train, dataset_name='gunpoint',target='label')

clf = TimeSeriesForestClassifier()
strategy = TSCStrategy(clf)

orchestrator = Orchestrator()
orchestrator.set_tasks([task])
orchestrator.set_strategies([strategy])
orchestrator.set_resampling(Single_Split())

orchestrator.run()