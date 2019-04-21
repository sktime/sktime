from sktime.experiments.orchestrator import Orchestrator
from sktime.highlevel import Task, TSCStrategy
from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sklearn.model_selection import KFold

from sktime.datasets import load_gunpoint
import pandas as pd



train = load_gunpoint(split='TRAIN')
test = load_gunpoint(split='TEST')
data = pd.concat([train,test], axis=0)

task = Task(case='TSC', data=train, target='class_val')
clf = TimeSeriesForestClassifier()
strategy = TSCStrategy(clf)

orchestrator = Orchestrator()
kf = KFold(n_splits=2)

orchestrator.run(tasks=[task], datasets=[data], strategies=[strategy], cv=kf)

