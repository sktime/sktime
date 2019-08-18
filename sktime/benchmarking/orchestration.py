__all__ = ["Orchestrator"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

import numpy as np
from sktime.highlevel.tasks import TSCTask, TSRTask


class Orchestrator:
    """
    Orchestrates the sequencing of running the machine learning experiments.

    Parameters
    ----------
    tasks: sktime.highlevel.Task
        task object
    datasets: pandas dataframe
        datasets in pandas skitme format
    strategies: list of sktime strategy
        strategy as per sktime.highlevel
    cv: sklearn.model_selection cross validation
        sklearn cross validation method. Must implement split()
    results: sktime result class
        Object for saving the results
    """

    def __init__(self, tasks, datasets, strategies, cv, results):

        self._validate_tasks_and_datasets(tasks, datasets)
        self.tasks = tasks
        self.datasets = datasets

        # validate strategies
        self._validate_strategy_names(strategies)
        self.strategies = strategies

        self.cv = cv
        self.results = results

    def _iter(self):
        """Iterator for orchestration"""
        # TODO skip data loading if all strategies are skipped in case results already exist

        for task, dataset in zip(self.tasks, self.datasets):
            data = dataset.load()  # load data into memory from dataset hook
            for strategy in self.strategies:
                for cv_fold, (train_idx, test_idx) in enumerate(self.cv.split(data)):
                    yield task, dataset, data, strategy, cv_fold, train_idx, test_idx

    def fit(self, overwrite_fitted_strategies=False, verbose=0):
        """Fit strategies on datasets"""

        for task, dataset, data, strategy, cv_fold, train_idx, test_idx in self._iter():

            # skip strategy, if overwrite is set to False and fitted strategy already exists
            if not overwrite_fitted_strategies and self.results.check_fitted_strategy_exists(
                    strategy, data.dataset_name):
                continue

            # else fit and save fitted strategy
            else:
                train = data.iloc[train_idx]
                strategy.fit(task, train)
                self.results.save_fitted_strategy(strategy=strategy,
                                                  dataset_name=data.dataset_name,
                                                  cv_fold=cv_fold)

    def predict(self, overwrite_predictions=False, predict_on_train=False, verbose=0):
        """Predict from saved fitted strategies"""
        raise NotImplementedError("Predicting from saved fitted strategies is not implemented yet")

    def fit_predict(self,
                    overwrite_predictions=False,
                    predict_on_train=False,
                    save_fitted_strategies=True,
                    overwrite_fitted_strategies=False,
                    verbose=True):
        """Fit and predict"""

        # check that for fitted strategies overwrite option is only set when save option is set
        if overwrite_fitted_strategies and not save_fitted_strategies:
            raise ValueError(f"Can only overwrite fitted strategies "
                             f"if fitted strategies are saved, but found: "
                             f"overwrite_fitted_strategies={overwrite_fitted_strategies} and"
                             f"save_fitted_strategies={save_fitted_strategies}")

        # fitting and prediction
        for task, dataset, data, strategy, cv_fold, train_idx, test_idx in self._iter():

            # check which results already exist
            train_pred_exist = self.results.check_predictions_exist(strategy.name, dataset.name,
                                                                    cv_fold, train_or_test='train')
            test_pred_exist = self.results.check_predictions_exist(strategy.name, dataset.name,
                                                                   cv_fold, train_or_test='test')
            fitted_stategy_exists = self.results.check_fitted_strategy_exists(strategy.name, dataset.name,
                                                                              cv_fold)

            # skip if overwrite is set to False for both predictions and strategies and all results exist
            if not overwrite_predictions and not overwrite_fitted_strategies and test_pred_exist and \
                    train_pred_exist and fitted_stategy_exists:
                print(f"Skipping strategy {strategy.name} on CV-fold {cv_fold} of dataset {dataset.name}")
                continue

            # split data into training and test sets
            train = data.iloc[train_idx]
            test = data.iloc[test_idx]

            # fit strategy
            if verbose:
                print(f"Fitting strategy {strategy.name} on CV-fold {cv_fold} of dataset {dataset.name}")
            strategy.fit(task, train)

            # save fitted strategy if save fitted strategies is set to True and overwrite is set to True or the
            # fitted strategy does not already exist
            if save_fitted_strategies and (overwrite_fitted_strategies or not fitted_stategy_exists):
                self.results.save_fitted_strategy(strategy, dataset_name=dataset.name, cv_fold=cv_fold)

            # optionally, predict on training set if predict on train is set to True and and overwrite is set to True
            # or the predicted values do not already exist
            if predict_on_train and (overwrite_predictions or not train_pred_exist):
                if verbose:
                    print(f"Predict strategy {strategy.name} on CV-fold {cv_fold} of the training set of dataset"
                          f" {dataset.name}")
                y_true = train.loc[:, task.target]
                y_pred = strategy.predict(train)
                y_proba = self._predict_proba_one(strategy, task, train, y_true, y_pred)
                self.results.save_predictions(strategy_name=strategy.name,
                                              dataset_name=dataset.name,
                                              index=train_idx,
                                              y_true=y_true,
                                              y_pred=y_pred,
                                              y_proba=y_proba,
                                              cv_fold=cv_fold,
                                              train_or_test='train')

            # predict on test set if overwrite predictions is set to True or predictions do not already exist
            if overwrite_predictions or not test_pred_exist:
                if verbose:
                    print(f"Predict strategy {strategy.name} on CV-fold {cv_fold} of dataset {dataset.name}")
                y_true = test.loc[:, task.target]
                y_pred = strategy.predict(test)
                y_proba = self._predict_proba_one(strategy, task, test, y_true, y_pred)
                self.results.save_predictions(dataset_name=dataset.name,
                                              strategy_name=strategy.name,
                                              index=test_idx,
                                              y_true=y_true,
                                              y_pred=y_pred,
                                              y_proba=y_proba,
                                              cv_fold=cv_fold,
                                              train_or_test='test')

        # save results as master file
        self.results.save()

    @staticmethod
    def _predict_proba_one(strategy, task, data, y_true, y_pred):
        """Predict strategy on one dataset"""
        # TODO always try to get probabilistic predictions first, compute deterministic predictions using
        #  argmax to avoid rerunning predictions, only if no predict_proba is available, run predict

        # if task is classification, return predicted probabilities for each class

        # if the task is classification and the strategies supports probabilistic predictions,
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
            #     y_proba[:, np.array(y_pred, dtype=np.int)] = 1

        else:
            return None

    @staticmethod
    def _validate_strategy_names(strategies):
        """Validate strategy names"""

        # Check uniqueness of strategy names
        names = [strategy.name for strategy in strategies]
        if not len(names) == len(set(names)):
            raise ValueError(f"Names of provided strategies are not unique: "
                             f"{names}")

        # Check for conflicts with estimator kwargs
        all_params = []
        for strategy in strategies:
            params = list(strategy.get_params(deep=False).keys())
            all_params.extend(params)

        invalid_names = set(names).intersection(set(all_params))
        if invalid_names:
            raise ValueError(f"Strategy names conflict with constructor "
                             f"arguments: {sorted(invalid_names)}")

        # Check for conflicts with double-underscore convention
        invalid_names = [name for name in names if "__" in name]
        if invalid_names:
            raise ValueError(f"Estimator names must not contain __: got "
                             f"{invalid_names}")

    @staticmethod
    def _validate_tasks_and_datasets(tasks, datasets):
        """Validate tasks"""
        # check input types
        if not isinstance(datasets, list):
            raise ValueError(f"datasets must be a list, but found: {type(datasets)}")
        if not isinstance(tasks, list):
            raise ValueError(f"tasks must be a list, but found: {type(tasks)}")

        # check if there is one task for each dataset
        if len(tasks) != len(datasets):
            raise ValueError("Inconsistent number of datasets and tasks, "
                             "there must be one task for each dataset")

        # check if task is either time series regression or classification, other tasks not supported yet
        if not all(isinstance(task, (TSCTask, TSRTask)) for task in tasks):
            raise NotImplementedError("Currently, only time series classification and time series "
                                      "regression tasks are supported")

        # check if all tasks are of the same type
        if not all(isinstance(task, type(tasks[0])) for task in tasks):
            raise ValueError("Not all tasks are of the same type")

