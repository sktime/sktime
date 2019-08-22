from sktime.benchmarking.orchestration import Orchestrator
from sktime.benchmarking.data import RAMDataset
from sktime.benchmarking.results import RAMResults
from sktime.benchmarking.evaluation import Evaluator
from sktime.benchmarking.metrics import Accuracy

from sktime.model_selection import SingleSplit
from sktime.highlevel.strategies import TSCStrategy
from sktime.highlevel.tasks import TSCTask
from sktime.classifiers.compose.ensemble import TimeSeriesForestClassifier

from sktime.benchmarking.evaluation import Evaluator

from sklearn.model_selection import KFold
import pandas as pd
import os
from sktime.datasets import load_arrow_head, load_gunpoint, load_italy_power_demand

# get path to the sktime datasets 
import sktime


# create the task and dataset objects manually for each dataset
dts_Italy = RAMDataset(dataset=load_italy_power_demand(), name='italy_power_demand')
task_Italy = TSCTask(target='class_val')

dts_ArrowHead = RAMDataset(dataset=load_arrow_head(), name='arrow_head')
task_ArrowHead = TSCTask(target='class_val')

dts_GunPoint = RAMDataset(dataset=load_gunpoint(), name='gunpoint')
task_GunPoint = TSCTask(target='class_val')

datasets=[dts_ArrowHead, dts_Italy, dts_GunPoint]
tasks=[task_ArrowHead, task_Italy, task_GunPoint]

clf = TimeSeriesForestClassifier(n_estimators=10, random_state=1)
strategy = TSCStrategy(clf)

resultRAM = RAMResults()

# run orchestrator
orchestrator = Orchestrator(datasets=datasets,
                            tasks=tasks,  
                            strategies=[strategy], 
                            cv=SingleSplit(random_state=1), 
                            results=resultRAM)
 
orchestrator.fit_predict(save_fitted_strategies=False)

# The results list can be obtained from loading the saved csv files by:
# results = RAMResults.load()

evaluator = Evaluator(resultRAM)

losses_df = evaluator.evaluate(metric=Accuracy())
print(evaluator.ranks())
print(losses_df)
# losses_df['Accuacy'] = 1- losses_df['loss']
# print(losses_df)