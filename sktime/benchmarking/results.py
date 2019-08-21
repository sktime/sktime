__all__ = ["HDDResults", "RAMResults", "UEAResults"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

import os

import numpy as np
import pandas as pd

from sktime.benchmarking.base import BaseResults, HDDBaseResults


class _ResultWrapper:
    """Single result class to ensure consistency for return object when loading results"""

    def __init__(self, strategy_name, dataset_name, index, y_true, y_pred, y_proba=None):
        # check input format
        if not all(isinstance(array, np.ndarray) for array in [y_true, y_pred]):
            raise ValueError(f"Prediction results have to stored as numpy arrays, "
                             f"but found: {[type(array) for array in [y_true, y_pred]]}")
        if not all(isinstance(name, str) for name in [strategy_name, dataset_name]):
            raise ValueError(f"Names must be strings, but found: "
                             f"{[type(name) for name in [strategy_name, dataset_name]]}")

        self.strategy_name = strategy_name
        self.dataset_name = dataset_name
        self.index = index
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba


class RAMResults(BaseResults):

    def __init__(self):
        self.results = {}
        super(RAMResults, self).__init__()

    def save_predictions(self, y_true, y_pred, y_proba, index, strategy_name=None, dataset_name=None,
                         train_or_test="test", cv_fold=0):
        key = f"{strategy_name}_{dataset_name}_{train_or_test}_{cv_fold}"
        predictions = np.column_stack([index, y_true, y_pred])
        self.results[key] = predictions
        self._append_names(strategy_name, dataset_name)

    def load_predictions(self, train_or_test="test", cv_fold=0):
        """Loads predictions for all datasets and strategies iteratively"""
        # TODO y_proba is currently ignored
        for dataset_name in self.dataset_names:
            for strategy_name in self.strategy_names:
                key = f"{strategy_name}_{dataset_name}_{train_or_test}_{cv_fold}"
                predictions = self.results[key]
                index = predictions[:, 0]
                y_true = predictions[:, 1]
                y_pred = predictions[:, 2]
                yield _ResultWrapper(strategy_name, dataset_name, index, y_true, y_pred)

    def check_predictions_exist(self, strategy, dataset_name, cv_fold, train_or_test="test"):
        # for in-memory results, always false
        return False

    def save_fitted_strategy(self, strategy, dataset_name, cv_fold):
        raise NotImplementedError()

    def load_fitted_strategy(self, strategy_name, dataset_name, cv_fold):
        raise NotImplementedError()

    def check_fitted_strategy_exists(self, strategy, dataset_name, cv_fold, train_or_test="test"):
        # for in-memory results, always false
        return False

    def save(self):
        # for in-memory results are currently not saved
        pass


class HDDResults(HDDBaseResults):

    def save_predictions(self, y_true, y_pred, y_proba, index, strategy_name=None, dataset_name=None,
                         train_or_test="test", cv_fold=0):
        """Save predictions"""
        # TODO y_proba is currently ignored
        path = self._make_file_path(self.path, strategy_name, dataset_name, cv_fold, train_or_test) + ".csv"
        results = pd.DataFrame({"index": index, "y_true": y_true, "y_pred": y_pred})
        results.to_csv(path, index=False, header=True)
        self._append_names(strategy_name, dataset_name)

    def load_predictions(self, train_or_test="test", cv_fold=0):
        """Load saved predictions"""

        for strategy_name in self.strategy_names:
            for dataset_name in self.dataset_names:
                path = self._make_file_path(self.path, strategy_name, dataset_name, cv_fold, train_or_test) + ".csv"
                results = pd.read_csv(path, header=0)
                index = results.loc[:, "index"].values
                y_true = results.loc[:, "y_true"].values
                y_pred = results.loc[:, "y_pred"].values
                yield _ResultWrapper(strategy_name, dataset_name, index, y_true, y_pred)

    def save_fitted_strategy(self, strategy, dataset_name, cv_fold):
        """Save fitted strategy"""
        path = self._make_file_path(self.fitted_strategies_path, strategy.name, dataset_name, cv_fold,
                                    train_or_test="train") + ".pickle"
        strategy.save(path)
        self._append_names(strategy.name, dataset_name)

    def load_fitted_strategy(self, strategy_name, dataset_name, cv_fold):
        """Load saved (fitted) strategy"""
        path = self._make_file_path(strategy_name, dataset_name, cv_fold)
        # TODO if we use strategy specific saving function, how do we remember how to load them? check file endings?
        raise NotImplementedError()

    def check_fitted_strategy_exists(self, strategy_name, dataset_name, cv_fold):
        path = self._make_file_path(self.fitted_strategies_path, strategy_name,
                                    dataset_name, cv_fold, train_or_test="train") + ".pickle"
        if os.path.isfile(path):
            return True
        else:
            return False

    def check_predictions_exist(self, strategy_name, dataset_name, cv_fold, train_or_test="test"):
        path = self._make_file_path(self.path, strategy_name, dataset_name, cv_fold, train_or_test) + ".csv"
        if os.path.isfile(path):
            return True
        else:
            return False

    @staticmethod
    def _make_file_path(path, strategy_name, dataset_name, cv_fold=0, train_or_test="test"):
        """Function to get paths for files, this basically encapsulate the storage logic of the class"""
        filepath = os.path.join(path, strategy_name, dataset_name)
        if not os.path.exists(filepath):
            # recursively create directory including intermediate-level folders
            os.makedirs(filepath)
        filename = f"{strategy_name}_{train_or_test}_{str(cv_fold)}"
        return os.path.join(filepath, filename)


class UEAResults(HDDBaseResults):
    pass
