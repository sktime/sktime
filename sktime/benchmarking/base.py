from abc import ABC, abstractmethod

__author__ = ["Markus Löning", "Viktor Kazakov"]
__all__ = ["BaseDataset", "HDDBaseDataset", "BaseResults", "HDDBaseResults"]

import os
from joblib import dump
from joblib import load


class BaseDataset:

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(name={self.name})"

    def load(self):
        raise NotImplementedError()

    @property
    def name(self):
        return self._name


class HDDBaseDataset(BaseDataset):

    def __init__(self, path, name):
        self._path = path
        super(HDDBaseDataset, self).__init__(name=name)

    @property
    def path(self):
        return self._path


class BaseResults:

    def __init__(self):
        # assigned during fitting of orchestration
        self.strategy_names = []
        self.dataset_names = []

    def save_predictions(self, y_true, y_pred, y_proba, index, strategy_name=None, dataset_name=None,
                         train_or_test="test", cv_fold=0):
        raise NotImplementedError()

    def load_predictions(self, train_or_test="test", cv_fold=0):
        """Loads predictions for all datasets and strategies iteratively"""
        raise NotImplementedError()

    def check_predictions_exist(self, strategy, dataset_name, cv_fold, train_or_test="test"):
        raise NotImplementedError()

    def save_fitted_strategy(self, strategy, dataset_name, cv_fold):
        raise NotImplementedError()

    def load_fitted_strategy(self, strategy_name, dataset_name, cv_fold):
        """Load fitted strategies for all datasets and strategies iteratively"""
        raise NotImplementedError()

    def check_fitted_strategy_exists(self, strategy, dataset_name, cv_fold):
        raise NotImplementedError()

    def _append_names(self, strategy_name, dataset_name):
        """Append names of datasets and strategies to results objects during orchestration"""
        if strategy_name not in self.strategy_names:
            self.strategy_names.append(strategy_name)

        if dataset_name not in self.dataset_names:
            self.dataset_names.append(dataset_name)

    def __repr__(self):
        class_name = self.__class__.__name__
        return f"{class_name}(strategies={self.strategy_names}, datasets={self.dataset_names})"

    def save(self):
        """Save results object as master file"""
        NotImplementedError()


class HDDBaseResults(BaseResults):

    def __init__(self, predictions_path, fitted_strategies_path=None):
        self._predictions_path = predictions_path
        self._fitted_strategies_path = predictions_path if fitted_strategies_path is None else fitted_strategies_path
        super(HDDBaseResults, self).__init__()

    @property
    def fitted_strategies_path(self):
        return self._fitted_strategies_path

    @property
    def path(self):
        return self._predictions_path

    def save(self):
        """Save results object as master file"""
        file = os.path.join(self.path, "results.pickle")

        # if file does not exist already, create a new one
        if not os.path.isfile(file):
            dump(self, file)

        # if file already exists, update file adding new datasets and strategies
        else:
            results = load(file)
            self.strategy_names = list(set(self.strategy_names + results.strategy_names))
            self.dataset_names = list(set(self.dataset_names + results.dataset_names))
            dump(self, file)


class BaseMetric(ABC):

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def calculate(self, y_true, y_pred):
        """
        Main method for performing the calculations.

        Parameters
        ----------
        y_true: array
            True dataset labels.
        y_pred: array
            predicted labels.

        Returns
        -------
        float
            Returns the result of the metric.
        """

    @abstractmethod
    def calculate_per_dataset(self, y_true, y_pred):
        """
        Calculates the loss per dataset

        Parameters
        ----------
        y_true: array
            True dataset labels.
        y_pred: array:
            predicted labels.

        Returns
        -------
        float
            Returns the result of the metric.
        """
