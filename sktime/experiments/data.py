import os
from sktime.utils.load_data import load_from_tsfile_to_dataframe
import re
import pandas as pd
from sktime.highlevel import Task

class DataHolder:
    """
    Class for holdig the data, schema, resampling splits and metadata
    """
    def __init__(self, data, task, dataset_name):
        """
        Parameters
        ----------
        data: pandas DataFrame
                dataset in pandas format
        task: sktime task
            sktime task object
        dataset_name: string
            Name of the dataset
        """

        self._data=data
        self._task=task
        self._dataset_name=dataset_name
    
    @property
    def data(self):
        return self._data
    
    @property
    def task(self):
        return self._task
    
    @property
    def dataset_name(self):
        return self._dataset_name

    def set_resampling_splits(self, train_idx, test_idx):
        """
        Saves the train test indices after the data is resampled

        Parameters
        -----------
        train_idx: numpy array
            array with indices of the train set
        test_idx: numpy array
            array with indices of the test set
        """
        self._train_idx = train_idx
        self._test_idx = test_idx

class DataLoader:
    """
    Class for loading large amounts of datasets. Usefull for automation of experiments. All datasets need to be saved under the same directory
    """

    def __init__(self, dts_dir, train_test_exists=True):
        """
        Parameters
        ----------
        dts_dir : string
            root directory where the datasets are saved
        train_test_exists : Boolean
            If True the datasets are split with the suffix _TRAIN and _TEST
        """

        self._dts_dir = dts_dir
        self._train_test_exists = train_test_exists
        
        self._all_datasets = iter(os.listdir(dts_dir))

    def load_ts(self, task_type, load_train=None, load_test=None, task_target='target'):
        """
        Iterates and loads the next dataset

        Parameters
        ----------
        task_type : string
            TSC or TSR
        load_train : Boolean
            loads the train set
        load_test : Boolean
            loads the test set
        task_target : String
            Name of the DataFrame column containing the label
        Returns
        -------
        sktime.dataholder
            dataholder ready for running experiments
        """


        if self._train_test_exists:
            if (load_train is None) and (load_test is None):
                raise ValueError('Specify whether the train or the test set needs to be loaded')
        #get the next dataset in the list
        dts_name =  next(self._all_datasets)

        datasets = os.listdir(os.path.join(self._dts_dir, dts_name))

        if self._train_test_exists:
            re_train = re.compile('.*_TRAIN.ts')
            re_test = re.compile('.*_TEST.ts')

            #find index of test or train file
            if load_train:
                for dts in datasets:
                    if re_train.match(dts):
                        loaded_dts = load_from_tsfile_to_dataframe(os.path.join(self._dts_dir, dts_name, dts))

            if load_test:
                for dts in datasets:
                    if re_test.match(dts):
                        loaded_dts = load_from_tsfile_to_dataframe(os.path.join(self._dts_dir, dts_name, dts))
            df = pd.DataFrame()
            if len(loaded_dts) == 2:
                df = loaded_dts[0]
                y = loaded_dts[1]
                df[task_target] = y

            task = Task(case=task_type, dataset_name=dts_name, data=df, target=task_target)
        return task




class Result:
    """
    Class for storing the results of the orchestrator
    """
    def __init__(self, dataset_name, strategy_name, true_labels, predictions):
        """
        Parameters
        -----------
        dataset_name: string
            name of the dataset
        strategy_name: string
            name of strategy
        true_labels: array
            true labels 
        predictions: array
            predictions of estimator
        """

        self._dataset_name = dataset_name
        self._strategy_name = strategy_name
        self._true_labels = true_labels
        self._predictions = predictions

    
    @property
    def dataset_name(self):
        return self._dataset_name
    @property
    def strategy_name(self):
        return self._strategy_name
    @property
    def true_labels(self):
        return self._true_labels
    @property
    def predictions(self):
        return self._predictions
