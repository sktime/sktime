# -*- coding: utf-8 -*-
"""Benchmarking orchestration module."""
__all__ = ["Orchestrator"]
__author__ = ["viktorkaz", "mloning"]

import logging

import pandas as pd
from sklearn.base import clone

from sktime.benchmarking.tasks import TSCTask, TSRTask

log = logging.getLogger()
console = logging.StreamHandler()
log.addHandler(console)


class Orchestrator:
    """Fit and predict one or more estimators on one or more datasets."""

    def __init__(self, tasks, datasets, strategies, cv, results):
        # validate datasets and tasks
        self._validate_tasks_and_datasets(tasks, datasets)
        self.tasks = tasks
        self.datasets = datasets

        # validate strategies
        self._validate_strategy_names(strategies)
        self.strategies = strategies

        self.cv = cv
        self.results = results

        # attach cv iterator to results object
        self.results.cv = cv

        # progress trackers
        self.n_strategies = len(strategies)
        self.n_datasets = len(datasets)
        self._strategy_counter = 0
        self._dataset_counter = 0

    def _iter(self):
        """Orchestration iterator."""
        # TODO: check if datasets are skipped entirely because predictions
        #  already exists before loading data,
        #  maybe do a dry-run first to find out which datasets to skip?
        for task, dataset in zip(self.tasks, self.datasets):
            # update counters
            self._strategy_counter = 0
            self._dataset_counter += 1

            # load data into memory from dataset hook
            data = dataset.load()

            # get target in case stratified cross-validation is used
            y = data[task.target]

            for strategy in self.strategies:
                self._strategy_counter += 1  # update counter

                for cv_fold, (train_idx, test_idx) in enumerate(self.cv.split(data, y)):
                    # for each fold, clone strategy to avoid updating
                    # already fitted strategies
                    strategy = clone(strategy)

                    yield (task, dataset, data, strategy, cv_fold, train_idx, test_idx)

    def fit(self, overwrite_fitted_strategies=False, verbose=False):
        """Fit strategies on datasets."""
        for (
            task,
            dataset,
            data,
            strategy,
            cv_fold,
            train_idx,
            _test_idx,
        ) in self._iter():

            # skip strategy, if overwrite is set to False and fitted
            # strategy already exists
            if (
                not overwrite_fitted_strategies
                and self.results.check_fitted_strategy_exists(
                    strategy, data.dataset_name
                )
            ):
                log.warn(
                    f"Skipping strategy: {strategy.name} on CV-fold: "
                    f"{cv_fold} of dataset: {dataset.name}"
                )
                continue

            # else fit and save fitted strategy
            else:
                train = data.iloc[train_idx]
                self._print_progress(
                    dataset.name, strategy.name, cv_fold, "train", "fit", verbose
                )
                strategy.fit(task, train)
                self.results.save_fitted_strategy(
                    strategy=strategy, dataset_name=data.dataset_name, cv_fold=cv_fold
                )

    def predict(
        self, overwrite_predictions=False, predict_on_train=False, verbose=False
    ):
        """Predict from saved fitted strategies."""
        raise NotImplementedError(
            "Predicting from saved fitted strategies is not implemented yet"
        )

    def fit_predict(
        self,
        overwrite_predictions=False,
        predict_on_train=False,
        save_fitted_strategies=True,
        overwrite_fitted_strategies=False,
        verbose=False,
    ):
        """Fit and predict."""
        # check that for fitted strategies overwrite option is only set when
        # save option is set
        if overwrite_fitted_strategies and not save_fitted_strategies:
            raise ValueError(
                f"Can only overwrite fitted strategies "
                f"if fitted strategies are saved, but found: "
                f"overwrite_fitted_strategies="
                f"{overwrite_fitted_strategies} and "
                f"save_fitted_strategies="
                f"{save_fitted_strategies}"
            )

        # fitting and prediction
        for task, dataset, data, strategy, cv_fold, train_idx, test_idx in self._iter():

            # check which results already exist
            train_pred_exist = self.results.check_predictions_exist(
                strategy.name, dataset.name, cv_fold, train_or_test="train"
            )
            test_pred_exist = self.results.check_predictions_exist(
                strategy.name, dataset.name, cv_fold, train_or_test="test"
            )
            fitted_stategy_exists = self.results.check_fitted_strategy_exists(
                strategy.name, dataset.name, cv_fold
            )

            # skip if overwrite is set to False for both predictions and
            # strategies and all results exist
            if (
                not overwrite_predictions
                and test_pred_exist
                and (train_pred_exist or not predict_on_train)
                and not overwrite_fitted_strategies
                and (fitted_stategy_exists or not save_fitted_strategies)
            ):
                log.warn(
                    f"Skipping strategy: {strategy.name} on CV-fold: "
                    f"{cv_fold} of dataset: {dataset.name}"
                )
                continue

            # split data into training and test sets
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]

            # fit strategy
            self._print_progress(
                dataset.name, strategy.name, cv_fold, "train", "fit", verbose
            )
            fit_estimator_start_time = pd.Timestamp.now()
            strategy.fit(task, train)
            fit_estimator_end_time = pd.Timestamp.now()

            # save fitted strategy if save fitted strategies is set to True
            # and overwrite is set to True or the
            # fitted strategy does not already exist
            if save_fitted_strategies and (
                overwrite_fitted_strategies or not fitted_stategy_exists
            ):
                self.results.save_fitted_strategy(
                    strategy, dataset_name=dataset.name, cv_fold=cv_fold
                )

            # optionally, predict on training set if predict on train is set
            # to True and and overwrite is set to True
            # or the predicted values do not already exist
            if predict_on_train and (overwrite_predictions or not train_pred_exist):
                y_true = train.loc[:, task.target]
                predict_estimator_start_time = pd.Timestamp.now()
                y_pred = strategy.predict(train)
                predict_estimator_end_time = pd.Timestamp.now()

                y_proba = self._predict_proba_one(strategy, task, train, y_true, y_pred)
                self.results.save_predictions(
                    strategy_name=strategy.name,
                    dataset_name=dataset.name,
                    index=train_idx,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    cv_fold=cv_fold,
                    fit_estimator_start_time=fit_estimator_start_time,
                    fit_estimator_end_time=fit_estimator_end_time,
                    predict_estimator_start_time=predict_estimator_start_time,
                    predict_estimator_end_time=predict_estimator_end_time,
                    train_or_test="train",
                )

            # predict on test set if overwrite predictions is set to True or
            # predictions do not already exist
            if overwrite_predictions or not test_pred_exist:
                y_true = test.loc[:, task.target]
                predict_estimator_start_time = pd.Timestamp.now()
                y_pred = strategy.predict(test)
                predict_estimator_end_time = pd.Timestamp.now()

                y_proba = self._predict_proba_one(strategy, task, test, y_true, y_pred)
                self.results.save_predictions(
                    dataset_name=dataset.name,
                    strategy_name=strategy.name,
                    index=test_idx,
                    y_true=y_true,
                    y_pred=y_pred,
                    y_proba=y_proba,
                    cv_fold=cv_fold,
                    fit_estimator_start_time=fit_estimator_start_time,
                    fit_estimator_end_time=fit_estimator_end_time,
                    predict_estimator_start_time=predict_estimator_start_time,
                    predict_estimator_end_time=predict_estimator_end_time,
                    train_or_test="test",
                )

        # save results as master file
        self.results.save()

    @staticmethod
    def _predict_proba_one(strategy, task, data, y_true, y_pred):
        """Predict strategy on one dataset."""
        # TODO always try to get probabilistic predictions first, compute
        #  deterministic predictions using
        #  argmax to avoid rerunning predictions, only if no predict_proba
        #  is available, run predict

        # if the task is classification and the strategies supports
        # probabilistic predictions,
        # get probabilistic predictions
        if isinstance(task, TSCTask) and hasattr(strategy, "predict_proba"):
            return strategy.predict_proba(data)

            # otherwise, return deterministic predictions in expected format
            # else:
            #     n_class_true = len(np.unique(y_true))
            #     n_class_pred = len(np.unique(y_pred))
            #     n_classes = np.maximum(n_class_pred, n_class_true)
            #     n_predictions = len(y_pred)
            #     y_proba = (n_predictions, n_classes)
            #     y_proba = np.zeros(y_proba)
            #     y_proba[:, np.array(y_pred, dtype=int)] = 1

        else:
            return None

    @staticmethod
    def _validate_strategy_names(strategies):
        """Validate strategy names."""
        # Check uniqueness of strategy names
        names = [strategy.name for strategy in strategies]
        if not len(names) == len(set(names)):
            raise ValueError(
                f"Names of provided strategies are not unique: " f"{names}"
            )

        # Check for conflicts with estimator kwargs
        all_params = []
        for strategy in strategies:
            params = list(strategy.get_params(deep=False).keys())
            all_params.extend(params)

        invalid_names = set(names).intersection(set(all_params))
        if invalid_names:
            raise ValueError(
                f"Strategy names conflict with constructor "
                f"arguments: {sorted(invalid_names)}"
            )

        # Check for conflicts with double-underscore convention
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(
                f"Estimator names must not contain __: got " f"{invalid_names}"
            )

    @staticmethod
    def _validate_tasks_and_datasets(tasks, datasets):
        """Validate tasks."""
        # check input types
        if not isinstance(datasets, list):
            raise ValueError(f"datasets must be a list, but found: {type(datasets)}")
        if not isinstance(tasks, list):
            raise ValueError(f"tasks must be a list, but found: {type(tasks)}")

        # check if there is one task for each dataset
        if len(tasks) != len(datasets):
            raise ValueError(
                "Inconsistent number of datasets and tasks, "
                "there must be one task for each dataset"
            )

        # check if task is either time series regression or classification,
        # other tasks not supported yet
        if not all(isinstance(task, (TSCTask, TSRTask)) for task in tasks):
            raise NotImplementedError(
                "Currently, only time series classification and time series "
                "regression tasks are supported"
            )

        # check if all tasks are of the same type
        if not all(isinstance(task, type(tasks[0])) for task in tasks):
            raise ValueError("Not all tasks are of the same type")

    def _print_progress(
        self,
        dataset_name,
        strategy_name,
        cv_fold,
        train_or_test,
        fit_or_predict,
        verbose,
    ):
        """Print progress."""
        if verbose:
            fit_or_predict = fit_or_predict.capitalize()
            if train_or_test == "train" and fit_or_predict == "predict":
                on_train = " (training set)"
            else:
                on_train = ""

            n_splits = self.cv.get_n_splits() - 1  # zero indexing

            log.warn(
                f"strategy: {self._strategy_counter}/{self.n_strategies} - "
                f"{strategy_name} "
                f"on CV-fold: {cv_fold}/{n_splits} "
                f"of dataset: {self._dataset_counter}/{self.n_datasets} - "
                f"{dataset_name}{on_train}"
            )
