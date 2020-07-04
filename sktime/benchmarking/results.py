__all__ = ["HDDResults", "RAMResults"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

import os

import numpy as np
import pandas as pd
from joblib import load
from sktime.benchmarking.base import BaseResults
from sktime.benchmarking.base import HDDBaseResults
from sktime.benchmarking.base import _PredictionsWrapper


class RAMResults(BaseResults):

    def __init__(self):
        self.results = {}
        super(RAMResults, self).__init__()

    def save_predictions(self, strategy_name, dataset_name, y_true, y_pred,
                         y_proba, index, cv_fold,
                         train_or_test):
        key = self._generate_key(strategy_name, dataset_name, cv_fold,
                                 train_or_test)
        index = np.asarray(index)
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        y_proba = np.asarray(y_proba)
        self.results[key] = _PredictionsWrapper(strategy_name, dataset_name,
                                                index, y_true, y_pred, y_proba)
        self._append_key(strategy_name, dataset_name)

    def load_predictions(self, cv_fold, train_or_test):
        """Loads predictions for all datasets and strategies iteratively"""
        for strategy_name, dataset_name in self._iter():
            key = self._generate_key(strategy_name, dataset_name, cv_fold,
                                     train_or_test)
            yield self.results[key]

    def check_predictions_exist(self, strategy, dataset_name, cv_fold,
                                train_or_test):
        # for in-memory results, always false, results are always overwritten
        return False

    def save_fitted_strategy(self, strategy, dataset_name, cv_fold):
        raise NotImplementedError()

    def load_fitted_strategy(self, strategy_name, dataset_name, cv_fold):
        raise NotImplementedError()

    def check_fitted_strategy_exists(self, strategy, dataset_name, cv_fold):
        # for in-memory results, always false, results are always overwritten
        return False

    def save(self):
        # in-memory results are currently not persisted (i.e saved to the disk)
        pass

    def _generate_key(self, strategy_name, dataset_name, cv_fold,
                      train_or_test):
        """Function to get paths for files, this basically encapsulate the
        storage logic of the class"""
        return f"{strategy_name}_{dataset_name}_{train_or_test}_{str(cv_fold)}"


class HDDResults(HDDBaseResults):

    def save_predictions(self, strategy_name, dataset_name, y_true, y_pred,
                         y_proba, index, cv_fold,
                         train_or_test):
        """Save predictions"""
        # TODO y_proba is currently ignored
        key = self._generate_key(strategy_name, dataset_name, cv_fold,
                                 train_or_test) + ".csv"
        results = pd.DataFrame(
            {"index": index, "y_true": y_true, "y_pred": y_pred})
        results.to_csv(key, index=False, header=True)
        self._append_key(strategy_name, dataset_name)

    def load_predictions(self, cv_fold, train_or_test):
        """Load saved predictions"""

        for strategy_name, dataset_name in self._iter():
            key = self._generate_key(strategy_name, dataset_name, cv_fold,
                                     train_or_test=train_or_test) + ".csv"
            results = pd.read_csv(key, header=0)
            index = results.loc[:, "index"].values
            y_true = results.loc[:, "y_true"].values
            y_pred = results.loc[:, "y_pred"].values
            yield _PredictionsWrapper(strategy_name, dataset_name, index,
                                      y_true, y_pred)

    def save_fitted_strategy(self, strategy, dataset_name, cv_fold):
        """Save fitted strategy"""
        path = self._generate_key(strategy.name, dataset_name, cv_fold,
                                  train_or_test="train") + ".pickle"
        strategy.save(path)
        self._append_key(strategy.name, dataset_name)

    def load_fitted_strategy(self, strategy_name, dataset_name, cv_fold):
        """Load saved (fitted) strategy"""
        for strategy_name, dataset_name in self._iter():
            key = self._generate_key(strategy_name, dataset_name, cv_fold,
                                     train_or_test="train") + ".pickle"
            # TODO if we use strategy specific saving function, how do we
            #  remember how to load them? check file endings?
            return load(key)

    def check_fitted_strategy_exists(self, strategy_name, dataset_name,
                                     cv_fold):
        path = self._generate_key(strategy_name, dataset_name, cv_fold,
                                  train_or_test="train") + ".pickle"
        if os.path.isfile(path):
            return True
        else:
            return False

    def check_predictions_exist(self, strategy_name, dataset_name, cv_fold,
                                train_or_test):
        path = self._generate_key(strategy_name, dataset_name, cv_fold,
                                  train_or_test) + ".csv"
        if os.path.isfile(path):
            return True
        else:
            return False

    def _generate_key(self, strategy_name, dataset_name, cv_fold,
                      train_or_test):
        """Function to get paths for files, this basically encapsulate
        the storage logic of the class"""
        filepath = os.path.join(self.path, strategy_name, dataset_name)
        if not os.path.exists(filepath):
            # recursively create directory including intermediate-level folders
            os.makedirs(filepath)
        filename = f"{strategy_name}_{train_or_test}_{cv_fold}"
        return os.path.join(filepath, filename)
