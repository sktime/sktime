from sktime.experiments.orchestrator import Orchestrator
from sktime.highlevel import TSCTask, TSCStrategy
from sktime.classifiers.ensemble import TimeSeriesForestClassifier
from sktime.experiments.data import DatasetHDD
from sktime.model_selection import PredefinedSplit
from sktime.experiments.data import ResultHDD
from sktime.experiments.data import DatasetLoadFromDir
from sklearn.model_selection import KFold

#create the task and dataset objects manually
dts_ArrowHead = DatasetHDD(dataset_loc='data/datasets/ArrowHead', dataset_name='ArrowHead')
task_ArrowHead = TSCTask(target='target')

dts_Beef = DatasetHDD(dataset_loc='data/datasets/Beef', dataset_name='Beef')
task_Beef = TSCTask(target='target')

#or create them automatically
dts_loader = DatasetLoadFromDir(root_dir='data/datasets')
datasets = dts_loader.load_datasets()
tasks = [TSCTask(target='target')] * len(datasets)

#create strategies
clf = TimeSeriesForestClassifier()
strategy = TSCStrategy(clf)

#result backend
result = ResultHDD(results_save_dir='data/results', strategies_save_dir='data/trained_strategies')


orchestrator = Orchestrator(datasets=datasets,
                            tasks=tasks,  
                            strategies=[strategy], 
                            cv=KFold(n_splits=10), #PredefinedSplit(), 
                            result=result)


orchestrator.run()


