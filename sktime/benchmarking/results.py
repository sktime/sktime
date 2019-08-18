__all__ = ["Result", "ResultsCSV", "ResultsRAM", "ResultsUEA"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

import os
import pandas as pd
from joblib import dump, load

from sktime.benchmarking.base import BaseResults


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
            Cross validation cv_fold
        """
        self._dataset_name = dataset_name
        self._strategy_name = strategy_name
        self._y_true = y_true
        self._y_pred = y_pred
        self._actual_probas = actual_probas
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


class ResultsRAM:
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
            Cross validation cv_fold
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


class ResultsUEA(BaseResults):
    pass


class ResultsCSV(BaseResults):

    def save_predictions(self, y_true, y_pred, y_proba, index, strategy_name=None, dataset_name=None,
                         train_or_test="test", cv_fold=0):
        """Save predictions"""
        # TODO y_proba is currently ignored
        path = self._make_file_path(self.path, strategy_name, dataset_name, cv_fold) + ".csv"
        results = pd.DataFrame({"index": index, "y_true": y_true, "y_pred": y_pred})
        results.to_csv(path, index=False, header=True)
        self._append_names(strategy_name, dataset_name)

    def load_predictions(self, train_or_test="test", cv_fold=0):
        """Load saved predictions"""

        for strategy_name in self.strategy_names:
            for dataset_name in self.dataset_names:
                filedir = os.path.join(self.path, strategy_name, dataset_name)
                filename = train_or_test + str(cv_fold) + ".csv"

                results = pd.read_csv(os.path.join(filedir, filename), header=0)
                index = results.loc[:, "index"]
                y_true = results.loc[:, "y_true"]
                y_pred = results.loc[:, "y_pred"]

                yield strategy_name, dataset_name, index, y_true, y_pred

    def save_fitted_strategy(self, strategy, dataset_name, cv_fold):
        """Save fitted strategy"""
        path = self._make_file_path(self.fitted_strategies_path, strategy.name, dataset_name, cv_fold) + ".pickle"
        strategy.save(path)
        self._append_names(strategy.name, dataset_name)

    def load_fitted_strategy(self, strategy_name, dataset_name, cv_fold):
        """Load saved (fitted) strategy"""
        path = self._make_file_path(strategy_name, dataset_name, cv_fold)
        # TODO if we use strategy specific saving function, how do we know how to load them? check file endings?
        raise NotImplementedError()

    def check_fitted_strategy_exists(self, strategy_name, dataset_name, cv_fold, train_or_test='test'):
        path = self._make_file_path(self.fitted_strategies_path, strategy_name, dataset_name, cv_fold) + ".pickle"
        if os.path.isfile(path):
            return True
        else:
            return False

    def check_predictions_exist(self, strategy_name, dataset_name, cv_fold, train_or_test='test'):
        path = self._make_file_path(self.path, strategy_name, dataset_name, cv_fold) + ".csv"
        if os.path.isfile(path):
            return True
        else:
            return False

    @staticmethod
    def _make_file_path(path, strategy_name, dataset_name, cv_fold):
        """Function to get paths for files"""
        filepath = os.path.join(path, strategy_name, dataset_name)
        if not os.path.exists(filepath):
            # recursively create directory including intermediate-level folders
            os.makedirs(filepath)
        filename = strategy_name + str(cv_fold)
        return os.path.join(filepath, filename)

    def _append_names(self, strategy_name, dataset_name):
        """Helper function to append names of datasets and strategies to results objects during orchestration"""
        if strategy_name not in self.strategy_names:
            self.strategy_names.append(strategy_name)

        if dataset_name not in self.dataset_names:
            self.dataset_names.append(dataset_name)

    def save(self):
        """Save results object as master file"""
        file = os.path.join(self.path, "results.pickle")

        # if file does not exist already, create a new one
        if not os.path.isfile(file):
            dump(self, file)

        # if file already exists, update file with new methods
        else:
            results = load(file)
            self.strategy_names = list(set(self.strategy_names + results.strategy_names))
            self.dataset_names = list(set(self.dataset_names + results.dataset_names))
            dump(self, file)
