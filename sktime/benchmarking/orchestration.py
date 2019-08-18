__all__ = ["Orchestrator"]
__author__ = ["Viktor Kazakov", "Markus Löning"]

import numpy as np


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

        # check if there is one task for each dataset
        if len(tasks) != len(datasets.names):
            raise ValueError("Inconsistent number of datasets and tasks, "
                             "there must be one task for each dataset")
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
        for task, data in zip(self.tasks, self.datasets):
            dts_loaded = data.load_predictions()
            for strategy in self.strategies:
                for cv_fold, (train, test) in enumerate(self.cv.split(dts_loaded)):
                    yield task, data, dts_loaded, strategy, cv_fold, train, test

    def fit(self, overwrite_fitted_strategies=False):
        """Fit strategies on datasets"""

        for task, data, dts_loaded, strategy, cv_fold, train, test in self._iter():

            # skip strategy, if overwrite is set to False and fitted strategy already exists
            if not overwrite_fitted_strategies and self.results.check_fitted_strategy_exists(strategy,
                                                                                             data.dataset_name):
                continue

            # else fit and save fitted strategy
            else:
                data_train = dts_loaded.iloc[train]
                strategy.fit(task, data_train)
                strategy.save_predictions(dataset_name=data.dataset_name,
                                          cv_fold=cv_fold,
                                          strategies_save_dir=self.results.strategies_save_dir)

    def predict(self, overwrite_predictions=False, predict_on_train=False):
        """Predict from saved fitted strategies"""
        # check if strategies have been saved
        # for task, data, dts_loaded, strategy, cv_fold, train, test in self._iter():
        # strategy = load(strategy)
        # predict
        raise NotImplementedError("Predicting from saved strategies is not implemented yet")

    def fit_predict(self,
                    overwrite_predictions=False,
                    predict_on_train=False,
                    save_fitted_strategies=True,
                    overwrite_fitted_strategies=False):
        """Fit and predict"""

        # check that for fitted strategies overwrite option is only set when save option is set
        if overwrite_fitted_strategies and not save_fitted_strategies:
            raise ValueError(f"Can only overwrite fitted strategies "
                             f"if fitted strategies are saved, but found: "
                             f"overwrite_fitted_strategies={overwrite_fitted_strategies} and"
                             f"save_fitted_strategies={save_fitted_strategies}")

        for task, data, dts_loaded, strategy, cv_fold, train, test in self._iter():

            # skip strategy if overwrite is set to False, no training set predictions are needed,
            # and test set predictions already exist
            if not overwrite_predictions and not predict_on_train and self.results.check_predictions_exists(
                    strategy, data.dataset_name, cv_fold, train_or_test='test'):
                continue

            # also skip strategy if overwrite is set to False, training set predictions are needed but already exist
            if not overwrite_predictions and predict_on_train and self.results.check_predictions_exists(
                    strategy, data.dataset_name, cv_fold, train_or_test='train'):
                continue

            # get train and test data
            data_train = dts_loaded.iloc[train]
            data_test = dts_loaded.iloc[test]

            # fit strategy
            strategy.fit(task, data_train)

            # save fitted strategies
            if save_fitted_strategies:
                if overwrite_fitted_strategies or not self.results.check_fitted_strategy_exists(strategy,
                                                                                                data.dataset_name):
                    strategy.save_predictions(dataset_name=data.dataset_name,
                                              cv_fold=cv_fold,
                                              strategies_save_dir=self.results.strategies_save_dir)

            # optionally, predict on training set
            if predict_on_train:
                y_pred = strategy.predict(data_train)
                y_train = data_train.loc[:, task.target]
                self.results.save_predictions(dataset_name=data.dataset_name,
                                              strategy_name=strategy.name,
                                              y_true=y_train,
                                              y_pred=y_pred,
                                              cv_fold=cv_fold,
                                              train_or_test='train')

            # predict on test set
            y_pred = strategy.predict(data_test)
            y_true = data_test.loc[:, task.target]

            # TODO always try to get probabilistic predictions, compute deterministic predictions using
            #  argmax to avoid rerunning predictions, only if no predict_proba is available, run predict
            # if available, get probabilistic predictions
            if hasattr(strategy, "predict_proba"):
                actual_probas = strategy.predict_proba(dts_loaded.iloc[test])
            else:
                # if no prediction probabilities were given set the probability of
                # the predicted class to 1 and the rest to zero.
                num_class_true = np.max(y_true) + 1
                num_class_pred = np.max(y_pred) + 1
                num_classes = max(num_class_pred, num_class_true)
                num_predictions = len(y_pred)
                actual_probas = (num_predictions, num_classes)
                actual_probas = np.zeros(actual_probas)
                actual_probas[np.arange(num_predictions), y_pred] = 1

            # save predictions
            self.results.save_predictions(dataset_name=data.dataset_name,
                                          strategy_name=strategy.name,
                                          y_true=y_true.tolist(),
                                          y_pred=y_pred.tolist(),
                                          actual_probas=actual_probas,
                                          cv_fold=cv_fold)

        # save results master file
        self.results.save()

    def _save_fitted_strategy(self, strategy, dataset_name, overwrite_fitted_strategy=False):
        """Save fitted strategies"""

        # if overwrite is set, simply save fitted strategies
        if overwrite_fitted_strategy:
            self.results.save_fitted_strategy(strategy, dataset_name)

        # else check if fitted strategy exists, if so, skip it, else save it
        if not self.results.check_fitted_strategy_exists(strategy, dataset_name):
            self.results.save_fitted_strategy(strategy, dataset_name)

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
