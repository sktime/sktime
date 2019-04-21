from sktime.experiments.orchestrator import Orchestrator
from sktime.highlevel import Task, TSCStrategy
from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sktime.experiments.data import DatasetHDD
from sktime.model_selection import SKtime_PredefinedSingleSplit
from sktime.datasets import load_gunpoint
import pandas as pd


dts_ArrowHead = DatasetHDD(dataset_loc='data/datasets/ArrowHead', dataset_name='ArrowHead', target='target')
task_ArrowHead = Task(case='TSC', data=dts_ArrowHead, target='target')

dts_Beef = DatasetHDD(dataset_loc='data/datasets/Beef', dataset_name='Beef')
task_Beef = Task(case='TSC', data=dts_Beef, target='target')

clf = TimeSeriesForestClassifier()
strategy = TSCStrategy(clf)

orchestrator = Orchestrator()

predef_split = SKtime_PredefinedSingleSplit(dataset_loc='data/datasets', train_suffix='_TRAIN.ts', test_suffix='_TEST.ts')

orchestrator.run(tasks=[task_ArrowHead, task_Beef], datasets=[dts_ArrowHead, dts_Beef], strategies=[strategy], cv=predef_split)

