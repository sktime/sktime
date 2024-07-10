"""Benchmarking base module."""

__author__ = ["mloning", "viktorkaz"]
__all__ = ["BaseDataset", "HDDBaseDataset", "BaseResults", "HDDBaseResults"]

import os
from abc import abstractmethod
from warnings import warn

import numpy as np
from joblib import dump, load


class BaseDataset:
    """Base dataset class."""

    def __init__(self, name):
        self._name = name

    def __repr__(self):
        """Repr dunder."""
        class_name = self.__class__.__name__
        return f"{class_name}(name={self.name})"

    def load(self):
        """Load dataset."""
        raise NotImplementedError()

    @property
    def name(self):
        """Name of dataset."""
        return self._name


class HDDBaseDataset(BaseDataset):
    """HDD dataset."""

    def __init__(self, path, name):
        self._path = path
        super().__init__(name=name)

    @property
    def path(self):
        """Path to dataset."""
        return self._path

    @staticmethod
    def _validate_path(path):
        """Validate paths."""
        # check if path already exists
        if not os.path.exists(path):
            raise ValueError(f"No dataset found at path: {path}")


class BaseResults:
    """Base results class."""

    def __init__(self):
        # assigned during fitting of orchestration
        self.strategy_names = []
        self.dataset_names = []
        self.cv = None

    def save_predictions(
        self,
        strategy_name,
        dataset_name,
        y_true,
        y_pred,
        y_proba,
        index,
        cv_fold,
        train_or_test,
    ):
        """Save predictions."""
        raise NotImplementedError()

    def load_predictions(self, cv_fold, train_or_test):
        """Load predictions for all datasets and strategies iteratively."""
        raise NotImplementedError()

    def check_predictions_exist(self, strategy, dataset_name, cv_fold, train_or_test):
        """Check that predictions exist."""
        raise NotImplementedError()

    def save_fitted_strategy(self, strategy, dataset_name, cv_fold):
        """Save fitted strategy."""
        raise NotImplementedError()

    def load_fitted_strategy(self, strategy_name, dataset_name, cv_fold):
        """Load fitted strategies for all datasets and strategies iteratively."""
        raise NotImplementedError()

    def check_fitted_strategy_exists(self, strategy, dataset_name, cv_fold):
        """Check that fitted strategy exists."""
        raise NotImplementedError()

    def _append_key(self, strategy_name, dataset_name):
        """Append names of datasets, strategies to results objects."""
        if strategy_name not in self.strategy_names:
            self.strategy_names.append(strategy_name)

        if dataset_name not in self.dataset_names:
            self.dataset_names.append(dataset_name)

    def _generate_key(self, strategy_name, dataset_name, cv_fold, train_or_test):
        raise NotImplementedError()

    def __repr__(self):
        """Representation dunder."""
        class_name = self.__class__.__name__
        return (
            f"{class_name}(strategies={self.strategy_names}, datasets="
            f"{self.dataset_names}, "
            f"cv_folds={self.cv.get_n_splits()})"
        )

    def save(self):
        """Save results object as master file."""
        NotImplementedError()

    def _iter(self):
        """Iterate over registry of results object."""
        for strategy_name in self.strategy_names:
            for dataset_name in self.dataset_names:
                yield strategy_name, dataset_name


class HDDBaseResults(BaseResults):
    """HDD results."""

    def __init__(self, path):
        # validate paths
        self._validate_path(path)

        # set path
        self._path = path

        super().__init__()

    @property
    def path(self):
        """Path for results on HDD."""
        return self._path

    def save(self):
        """Save results object as master file."""
        file = os.path.join(self.path, "results.pickle")

        # if file does not exist already, create a new one
        if not os.path.isfile(file):
            dump(self, file)

        # if file already exists, update file adding new datasets,
        # strategies and/or cv_folds
        else:
            results = load(file)
            self.strategy_names = list(
                set(self.strategy_names + results.strategy_names)
            )
            self.dataset_names = list(set(self.dataset_names + results.dataset_names))
            dump(self, file)

    @staticmethod
    def _validate_path(path):
        """Validate paths."""
        # check if path already exists
        if os.path.exists(path):
            if not os.path.isdir(path):
                raise ValueError("path already exists and is not a directory")

            elif os.path.isfile(os.path.join(path, "results.pickle")):
                warn(
                    f"Existing results file found in given path: {path}. "
                    f"Results file will be updated"
                )

            elif (
                len([file for file in os.listdir(path) if not file.startswith(".")]) > 0
            ):
                warn("path already exists and is not empty")


class _PredictionsWrapper:
    """Single result class for consistency of return object when loading results."""

    def __init__(
        self,
        strategy_name,
        dataset_name,
        index,
        y_true,
        y_pred,
        fit_estimator_start_time,
        fit_estimator_end_time,
        predict_estimator_start_time,
        predict_estimator_end_time,
        y_proba=None,
    ):
        # check input format
        if not all(isinstance(array, np.ndarray) for array in [y_true, y_pred]):
            raise ValueError(
                f"Prediction results have to stored as numpy arrays, "
                f"but found: {[type(array) for array in [y_true, y_pred]]}"
            )
        if not all(isinstance(name, str) for name in [strategy_name, dataset_name]):
            raise ValueError(
                f"Names must be strings, but found: "
                f"{[type(name) for name in [strategy_name, dataset_name]]}"
            )

        self.strategy_name = strategy_name
        self.dataset_name = dataset_name
        self.index = index
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.fit_estimator_start_time = fit_estimator_start_time
        self.fit_estimator_end_time = fit_estimator_end_time
        self.predict_estimator_start_time = predict_estimator_start_time
        self.predict_estimator_end_time = predict_estimator_end_time


class BaseMetric:
    def __init__(self, name, **kwargs):
        self.name = name
        self.kwargs = kwargs

    @abstractmethod
    def compute(self, y_true, y_pred):
        """Compute mean and standard error of metric."""
