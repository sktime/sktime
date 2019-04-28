import os
from sktime.utils.load_data import load_from_tsfile_to_dataframe
from sktime.utils.results_writing import write_results_to_uea_format
from sktime.highlevel import TSCTask, ForecastingTask
import re
import pandas as pd
from abc import ABC
import pickle
import numpy as np
import csv
class DatasetHDD:
    """
    Another class for holding the data
    """
    def __init__(self, dataset_loc, dataset_name, train_test_exists=True, sufix_train='_TRAIN.ts', suffix_test='_TEST.ts' ,target='target'):
        """
        Parameters
        ----------
        dataset_loc: string
            path on disk where the dataset is saved. Path to directory without the file name should be provided
        dataset_name: string
            Name of the dataset
        train_test_exists: Boolean
            flag whether the test train split already exists
        sufix_train: string
            train file suffix
        suffix_test: string
            test file suffix

        Returns
        -------
        pandas DataFrame:
            dataset in pandas DataFrame format
        """
        self._dataset_loc = dataset_loc
        self._dataset_name = dataset_name
        self._train_test_exists = train_test_exists
        self._target = target
        self._suffix_train = sufix_train
        self._suffix_test = suffix_test

    @property
    def dataset_name(self):
        return self._dataset_name

    def load(self):
        #TODO curently only the current use case with saved datasets on the disk in a certain format is supported. This should be made more general.
        if self._train_test_exists:
   

            loaded_dts_train = load_from_tsfile_to_dataframe(os.path.join(self._dataset_loc, self._dataset_name + self._suffix_train))
            loaded_dts_test = load_from_tsfile_to_dataframe(os.path.join(self._dataset_loc, self._dataset_name + self._suffix_test))

            data_train = loaded_dts_train[0]
            y_train = loaded_dts_train[1]

            data_test = loaded_dts_test[0]
            y_test = loaded_dts_test[1]

            #concatenate the two dataframes
            data_train[self._target] = y_train
            data_test[self._target] = y_test

            data = pd.concat([data_train,data_test], axis=0, keys=['train','test']).reset_index(level=1, drop=True)

            return data

class DatasetLoadFromDir:
    """
    Loads all datasets in a root directory
    """
    def __init__(self, root_dir):
        """
        Parameters
        ----------
        root_dir: string
            Root directory where the datasets are located
        """
        self._root_dir = root_dir
    
    def load_datasets(self, train_test_exists=True):
        """
        Parameters
        ----------
        train_test_exists: Boolean
            Flag whether the test/train split exists

        Returns
        -------
        DatasetHDD:
            list of DatasetHDD objects
        """
        datasets = os.listdir(self._root_dir)

        data = []
        for dts in datasets:
            dts = DatasetHDD(dataset_loc=os.path.join(self._root_dir, dts), dataset_name=dts, train_test_exists=train_test_exists)
            data.append(dts)
        return data
    
class Result:
    """
    Used for passing results to the analyse results class
    """

    def __init__(self,dataset_name, strategy_name, y_true, y_pred):
        """
        Parameters
        ----------
        dataset_name: string
            Name of the dataset
        strategy_name: string
            name of the strategy
        y_true: list
            True labels
        y_pred: list
            predictions
        """
        self._dataset_name = dataset_name
        self._strategy_name = strategy_name
        self._y_true = y_true
        self._y_pred = y_pred

    @property
    def dataset_name(self):
        return self._dataset_name
    
    @property
    def strategy_name(self):
        return self._strategy_name
    
    @property
    def y_true(self):
        return self._y_true

    @property
    def y_pred(self):
        return self._y_pred
# class DataHolder:
#     """
#     Class for holdig the data, schema, resampling splits and metadata
#     """
#     def __init__(self, data, task, dataset_name):
#         """
#         Parameters
#         ----------
#         data: pandas DataFrame
#             dataset in pandas format
#         task: sktime task
#             sktime task object
#         dataset_name: string
#             Name of the datasets
#         """

#         self._data=data
#         self._task=task
#         self._dataset_name=dataset_name
    
#     @property
#     def data(self):
#         return self._data
    
#     @property
#     def task(self):
#         return self._task
    
#     @property
#     def dataset_name(self):
#         return self._dataset_name

#     def set_resampling_splits(self, train_idx, test_idx):
#         """
#         Saves the train test indices after the data is resampled

#         Parameters
#         -----------
#         train_idx: numpy array
#             array with indices of the train set
#         test_idx: numpy array
#             array with indices of the test set
#         """
#         self._train_idx = train_idx
#         self._test_idx = test_idx

# class DataLoader:
#     """
#     Class for loading large amounts of datasets. Usefull for automation of experiments. All datasets need to be saved under the same directory
#     """

#     def __init__(self, dts_dir, task_types, train_test_exists=True):
#         """
#         Parameters
#         ----------
#         dts_dir : string
#             root directory where the datasets are saved
#         train_test_exists : Boolean
#             If True the datasets are split with the suffix _TRAIN and _TEST
#         task_types : string
#             TSC or TSR
#         """

#         self._dts_dir = dts_dir
#         self._train_test_exists = train_test_exists
#         self._task_types = task_types
        
#         self._all_datasets = iter(os.listdir(dts_dir))

#     def _load_ts(self, load_train=None, load_test=None, task_target='target'):
#         """
#         Iterates and loads the next dataset

#         Parameters
#         ----------
        
#         load_train : Boolean
#             loads the train set
#         load_test : Boolean
#             loads the test set
#         task_target : String
#             Name of the DataFrame column containing the label
#         Returns
#         -------
#         tuple of sktime.dataholder
#             dataholder_train and dataholder_test.
#         """


#         if self._train_test_exists:
#             if (load_train is None) and (load_test is None):
#                 raise ValueError('At least the train or the test set needs to be loaded')
#         #get the next dataset in the list
#         dts_name =  next(self._all_datasets)

#         datasets = os.listdir(os.path.join(self._dts_dir, dts_name))

#         dataholder_train = None
#         dataholder_test = None
        
#         if self._train_test_exists:
#             re_train = re.compile('.*_TRAIN.ts')
#             re_test = re.compile('.*_TEST.ts')

#             #find index of test or train file
#             if load_train:
#                 data = pd.DataFrame()

#                 for dts in datasets:
#                     if re_train.match(dts):
#                         loaded_dts = load_from_tsfile_to_dataframe(os.path.join(self._dts_dir, dts_name, dts))
#                         if len(loaded_dts) == 2:
#                             data = loaded_dts[0]
#                             y = loaded_dts[1]
#                             data[task_target] = y
#                             task = Task(case=self._task_types, dataset_name=dts_name, data=data, target=task_target)
#                             dataholder_train = DataHolder(data=data, task=task, dataset_name=dts_name)
#             if load_test:
#                 data  = pd.DataFrame()

#                 for dts in datasets:
#                     if re_test.match(dts):
#                         loaded_dts = load_from_tsfile_to_dataframe(os.path.join(self._dts_dir, dts_name, dts))
#                         if len(loaded_dts) == 2:
#                             data = loaded_dts[0]
#                             y = loaded_dts[1]
#                             data[task_target] = y
#                             task = Task(case=self._task_types, dataset_name=dts_name, data=data, target=task_target)
#                             dataholder_test = DataHolder(data=data, task=task, dataset_name=dts_name)
        
#         return dataholder_train, dataholder_test


#     def load(self, load_test_train='both', task_target='target'):
#         """
#         Method for loading sequentially the data irrespective of how it was supplied, i.e. ts files, csv, db.
        
#         Parameters
#         ----------
        
#         load_test_train : String
#             Acceptable valies: test, train, both. Loads either the test set, the train set or both.
#         load_test : Boolean
#             loads the test set
#         task_target : String
#             Name of the DataFrame column containing the label
#         Returns
#         -------
#         sktime.dataholder
#             dataholder ready for running experiments
#         """
#         #TODO expand as the list of backend functions grows

#         acceptable_test_train_values = ['test', 'train', 'both']
#         if load_test_train not in acceptable_test_train_values:
#             raise ValueError('Acceptable values for the load_test_train parameter are: test, train or both')
        
#         if load_test_train == 'train':
#             return self._load_ts(load_train=True, task_target=task_target)
#         if load_test_train == 'test':
#             return self._load_ts(load_test=True, task_target=task_target)
#         if load_test_train == 'both':
#             return self._load_ts(load_train=True, load_test=True, task_target=task_target)

class SKTimeResult(ABC):
    def save(self):
        """
        Saves the result
        """
    def save_trained_strategy(self):
        """
        method for persisting the trained strategies
        """
    def load(self):
        """
        method for loading the results
        """

class ResultHDD(SKTimeResult):
    """
    Class for storing the results of the orchestrator
    """
    def __init__(self, results_save_dir, strategies_save_dir=None):
        """
        Parameters
        -----------
        results_save_dir: string
            path where the results will be saved
        strategies_save_dir: string
            path where the strategies can be saved
        """

        self._results_save_dir = results_save_dir
        self._strategies_save_dir = strategies_save_dir

    def save(self, dataset_name, strategy_name, y_true, y_pred, cv_fold):
        if not os.path.exists(self._results_save_dir):
            os.makedirs(self._results_save_dir)
        #TODO BUG: write write_results_to_uea_format does not write the results property unless the probas are provided as well.
        #Dummy probas to make the write_results_to_uea_format function work
        y_true = list(map(int, y_true))
        y_pred = list(map(int, y_pred))
        num_class_true = np.max(y_true)
        num_class_pred = np.max(y_pred)
        num_classes = max(num_class_pred, num_class_true)
        num_predictions = len(y_pred)
        probas = (num_predictions, num_classes)
        probas = np.zeros(probas)

        write_results_to_uea_format(output_path=self._results_save_dir,
                                    classifier_name=strategy_name,
                                    dataset_name=dataset_name,
                                    actual_class_vals=y_true,
                                    predicted_class_vals=y_pred,
                                    actual_probas = probas,
                                    resample_seed=cv_fold)


    def save_trained_strategy(self, strategy, dataset_name, cv_fold):
        if self._strategies_save_dir is None:
            raise ValueError('Please provide a directory for saving the strategies')
        
        #TODO implement check for overwriting already saved files
        save_path = os.path.join(self._strategies_save_dir, dataset_name)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        #TODO pickling will not work for all strategies
        pickle.dump(strategy, open(os.path.join(save_path, strategy.name + 'cv_fold'+str(cv_fold)+ '.p'),"wb"))
    
    def load(self):
        """
        Returns
        -------
        list:
            sktime results
        """
        results = []
        for r,d,f in os.walk(self._results_save_dir):
            for file_to_load in f:
                if file_to_load.endswith(".csv"):
                    #found .csv file. Load it and create results object
                    path_to_load = os.path.join(r, file_to_load)
                    current_row= 0
                    strategy_name = ""
                    dataset_name = ""
                    y_true = []
                    y_pred = []
                    with open(path_to_load) as csvfile:
                        readCSV = csv.reader(csvfile, delimiter=',')
                        for row in readCSV:
                            if current_row == 0:
                                strategy_name = row[0]
                                dataset_name = row[1]
                                current_row +=1
                            elif current_row >=4:
                                y_true.append(row[0])
                                y_pred.append(row[1])
                                current_row +=1
                            else:
                                current_row +=1
                    #create result object and append
                    result = Result(dataset_name=dataset_name, strategy_name=strategy_name, y_true=y_true, y_pred=y_pred)
                    results.append(result)
        
        return results