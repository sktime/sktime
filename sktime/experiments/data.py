import csv
import os
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from ..utils.load_data import load_from_tsfile_to_dataframe
from ..utils.results_writing import write_results_to_uea_format
import logging

__all__ =['DatasetHDD','DatasetRAM','DatasetLoadFromDir','Result','ResultRAM','ResultHDD']
__author__ = ['Viktor Kazakov']
class DatasetHDD:
    """
    Another class for holding the data
    """

    def __init__(self, dataset_loc, dataset_name, train_test_exists=True, sufix_train='_TRAIN.ts',
                 suffix_test='_TEST.ts', target='target'):
        """
        Parameters
        ----------
        dataset_loc : str
            path on disk where the dataset is saved. Path to directory without the file name should be provided
        dataset_name : str
            Name of the dataset
        train_test_exists : bool
            flag whether the test train split already exists
        sufix_train : str
            train file suffix
        suffix_test : str
            test file suffix
        """
        self._dataset_loc = dataset_loc
        self._dataset_name = dataset_name
        self._train_test_exists = train_test_exists
        self._target = target
        self._suffix_train = sufix_train
        self._suffix_test = suffix_test

    @property
    def dataset_name(self):
        """
        Returns
        -------
        str
            Name of the dataset
        """
        return self._dataset_name

    def load(self):
        """
        Returns
        -------
        pandas DataFrame:
            dataset in pandas DataFrame format
        """
        # TODO curently only the current use case with saved datasets on the disk in a certain format is supported. This should be made more general.
        if self._train_test_exists:
            loaded_dts_train = load_from_tsfile_to_dataframe(
                os.path.join(self._dataset_loc, self._dataset_name + self._suffix_train))
            loaded_dts_test = load_from_tsfile_to_dataframe(
                os.path.join(self._dataset_loc, self._dataset_name + self._suffix_test))

            data_train = loaded_dts_train[0]
            y_train = loaded_dts_train[1]

            data_test = loaded_dts_test[0]
            y_test = loaded_dts_test[1]

            # concatenate the two dataframes
            data_train[self._target] = y_train
            data_test[self._target] = y_test

            data = pd.concat([data_train, data_test], axis=0, keys=['train', 'test']).reset_index(level=1, drop=True)

            return data


class DatasetRAM:
    def __init__(self, dataset, dataset_name):
        """
        Container for storing a dataset in memory

        Paramteters
        -----------
        dataset : pandas DataFrame
            The actual dataset
        dataset_name : str
            Name of the dataset
        """

        self._dataset = dataset
        self._dataset_name = dataset_name

    @property
    def dataset_name(self):
        """
        Returns
        -------
        str
            Name of the dataset
        """
        return self._dataset_name

    def load(self):
        """
        Returns
        -------
        pandas DataFrame
            dataset in pandas DataFrame format
        """
        return self._dataset


class DatasetLoadFromDir:
    """
    Loads all datasets in a root directory
    """

    def __init__(self, root_dir):
        """
        Parameters
        ----------
        root_dir : str
            Root directory where the datasets are located
        """
        self._root_dir = root_dir

    def load_datasets(self, train_test_exists=True):
        """
        Parameters
        ----------
        train_test_exists : bool
            Flag whether the test/train split exists

        Returns
        -------
        list
            list of DatasetHDD objects
        """
        datasets = os.listdir(self._root_dir)

        data = []
        for dts in datasets:
            dts = DatasetHDD(dataset_loc=os.path.join(self._root_dir, dts), dataset_name=dts,
                             train_test_exists=train_test_exists)
            data.append(dts)
        return data


class Result:
    """
    Used for passing results to the analyse results class
    """

    def __init__(self, dataset_name, strategy_name, y_true, y_pred, actual_probas, cv):
        """
        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        strategy_name : str
            name of the strategy
        y_true : list
            True labels
        actual_probas : array
             Probabilities for each class. Result of `estimator.predict_proba()`
        y_pred : list
            predictions
        cv_fold : int
            Cross validation fold
        """
        self._dataset_name = dataset_name
        self._strategy_name = strategy_name
        self._y_true = y_true
        self._y_pred = y_pred
        self._actual_probas=actual_probas
        self._cv = cv

    @property
    def dataset_name(self):
        """
        Returns
        -------
        str
            Name of the dataset
        """
        return self._dataset_name

    @property
    def strategy_name(self):
        """
        Returns
        -------
        str
            Name of the strategy
        """
        return self._strategy_name

    @property
    def y_true(self):
        """
        Returns
        -------
        list
            True target variables
        """
        return self._y_true

    @property
    def y_pred(self):
        """
        Returns
        -------
        list
            Predicted target variables
        """
        return self._y_pred


class SKTimeResult(ABC):
    @abstractmethod
    def save(self):
        """
        Saves the result
        """

    @abstractmethod
    def load(self):
        """
        method for loading the results
        """


class ResultRAM(SKTimeResult):
    """
    Class for storing the results of the experiments in memory
    """

    def __init__(self):
        self._results = []

    def save(self, dataset_name, strategy_name, y_true, y_pred, actual_probas, cv_fold):
        """
        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        strategy_name : str
            Name of the strategy
        y_true : array
            True lables array
        y_pred: array
            Predictions array
        actual_probas : array
             Probabilities for each class. Result of `estimator.predict_proba()`
        cv_fold : int
            Cross validation fold
        """
        result = Result(dataset_name=dataset_name, 
                        strategy_name=strategy_name, 
                        y_true=y_true, 
                        y_pred=y_pred, 
                        actual_probas=actual_probas, 
                        cv=cv_fold)
        self._results.append(result)

    def load(self):
        """
        Returns
        -------
        list
            sktime results
        """
        return self._results


class ResultHDD(SKTimeResult):
    """
    Class for storing the results of the orchestrator
    """

    def __init__(self, results_save_dir, strategies_save_dir):
        """
        Parameters
        -----------
        results_save_dir : str
            path where the results will be saved
        strategies_save_dir : str
            path for saving the strategies
        """

        self._results_save_dir = results_save_dir
        self._strategies_save_dir = strategies_save_dir

    @property
    def strategies_save_dir(self):
        """
        Returns
        -------
        str
            Path where the strategies will be saved
        """
        return self._strategies_save_dir

    def save(self, dataset_name, strategy_name, y_true, y_pred, actual_probas, cv_fold):
        """
        Parameters
        ----------
        dataset_name : str
            Name of the dataset
        strategy_name : str
            Name of the strategy
        y_true : array
            True lables array
        y_pred : array
            Predictions array.
        actual_probas : array
             Probabilities for each class. Result of `estimator.predict_proba()`
        cv_fold : int
            Cross validation fold
        """
        if not os.path.exists(self._results_save_dir):
            os.makedirs(self._results_save_dir)
        
        
        write_results_to_uea_format(output_path=self._results_save_dir,
                                    classifier_name=strategy_name,
                                    dataset_name=dataset_name,
                                    actual_class_vals=y_true,
                                    predicted_class_vals=y_pred,
                                    actual_probas=actual_probas,
                                    resample_seed=cv_fold)

    def load(self):
        """
        Returns
        -------
        list
            sktime results
        """
        results = []
        for r, d, f in os.walk(self._results_save_dir):
            for file_to_load in f:
                if file_to_load.endswith(".csv"):
                    # found .csv file. Load it and create results object
                    path_to_load = os.path.join(r, file_to_load)
                    current_row = 0
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
                                current_row += 1
                            elif current_row >= 4:
                                y_true.append(row[0])
                                y_pred.append(row[1])
                                current_row += 1
                            else:
                                current_row += 1
                    # create result object and append
                    result = Result(dataset_name=dataset_name, strategy_name=strategy_name, y_true=y_true,
                                    y_pred=y_pred)
                    results.append(result)

        return results
