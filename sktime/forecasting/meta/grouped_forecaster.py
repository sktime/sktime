#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Implements model that fits different models for different clusters of timeseries."""

__author__ = ["boukepostma"]
__all__ = ["GroupedForecaster"]

import pandas as pd
from sklearn import clone

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection._tune import BaseGridSearch


def as_list(val):
    """
    Return a list of the input value.

    :param val: the input value.
    :returns: the input value as a list.

    :Example:

    >>> as_list('test')
    ['test']

    >>> as_list(['test1', 'test2'])
    ['test1', 'test2']
    """
    treat_single_value = str

    if isinstance(val, treat_single_value):
        return [val]

    if hasattr(val, "__iter__"):
        return list(val)

    return [val]


class GroupedForecaster(BaseForecaster):
    """
    Construct an estimator per data group.

    Splits data and aggregates time series into groups by values of a single column
    and fits one estimator per group.
    If the estimator is a gridsearch:
        1. First perform gridsearch per group.
        2. Each grid's best hyperparameters
        3. When reusing the model (by fitting or predicting): only use best models from
            step 2, don't refit grid search.

    :param estimator: the model/pipeline to be applied per group
    :param depth: integer defining the depth of aggregation:
                  0 is top-level, 1 is 1 split below, etc
    :param aggregations: dictionary mapping col names to aggregation methods/functions
    :param fallback_model: model to use when an error is raised during fitting of group
    """

    _tags = {
        "scitype:y": "both",
        "ignores-exogeneous-X": False,
        "y_inner_mtype": "pd.Series",
        "requires-fh-in-fit": False,
    }

    def __init__(
        self,
        estimator,
        depth,
        fallback_model=None,
    ):
        self.estimator = estimator
        self.depth = depth
        self.fallback_model = fallback_model
        super(GroupedForecaster, self).__init__()
        self.estimators_ = None
        self.grids = {}

    def __fit_single_ts(self, group, y, X=None, fh=None):
        try:
            X = None if X.empty else X
            if not self.estimators_ and isinstance(self.estimator, BaseGridSearch):
                estimator = clone(self.estimator).fit(y, X, fh)

                # Save grid to later access grid CV results
                self.grids[group] = estimator
                return estimator.best_forecaster_

            if self.estimators_ and isinstance(self.estimator, BaseGridSearch):
                return clone(self.estimators_[group]).fit(y, X, fh)

            else:
                return clone(self.estimator).fit(y, X, fh)
        except Exception as e:
            raise type(e)(f"Exception for time serie of group {group}: {e}")

    def _fit(self, y, X, fh=None):
        """
        Fit the model using X, y as training data.

        :param X: pd.DataFrame, shape=(n_columns, n_samples,)
                grouping columns and training data.
        :param y: pd.Series, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        self.fallback_ = None

        if self.fallback_model:
            self.fallback_ = clone(self.fallback_model).fit(y, X, fh)

        # y needs to come in X as a workaround for sktime's current data checks
        y = X.filter(regex="^y")
        X = X.filter(regex="^(?!y).*")

        # Get data from desired depth-level
        y_cols_to_predict = [col for col in y.columns if col.count("__") == self.depth]
        X_cols_to_predict = [col for col in X.columns if col.count("__") == self.depth]
        X = X.loc[:, X_cols_to_predict]

        self.estimators_ = {
            # Fit a clone of the transformer to each time serie
            group: self.__fit_single_ts(
                group, y.loc[:, group], X.loc[:, X.columns.str.endswith(group[1:])], fh
            )
            for group in y_cols_to_predict
        }

        self.groups_ = as_list(self.estimators_.keys())

        return self

    def __predict_single_ts(self, group, fh=None, X=None):
        """Predict a single group by getting its estimator from the fitted dict."""
        try:
            group_predictor = self.estimators_[group]
        except KeyError:
            if self.fallback_:
                group_predictor = self.fallback_
            else:
                raise ValueError(
                    f"Found new group {group} during predict with"
                    "fallback_model = None"
                )
        X = None if X.empty else X
        return pd.DataFrame(group_predictor.predict(X=X, fh=fh))

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """
        Predict for all groups on new data.

        :param fh: XXX
        :param X: XXX
        :return: array, shape=(n_samples,) the predicted data
        """
        if return_pred_int:
            raise NotImplementedError()
        # y needs to come in X as a workaround for sktime's current data checks
        X = X.filter(regex="^(?!y).*")

        # Get data from desired depth-level
        X_cols_to_use = [col for col in X.columns if col.count("__") == self.depth]
        X = X.loc[:, X_cols_to_use]

        return pd.concat(
            [
                self.__predict_single_ts(
                    group, fh, X.loc[:, X.columns.str.endswith(group[1:])]
                )
                for group in self.groups_
            ],
            axis=1,
        ).sum(axis=1)

    def _get_cv_results_agg(self, y, X, scoring=None, cv=None):
        if not isinstance(self.estimator, BaseGridSearch):
            raise ValueError("forecaster should be a BaseGridSearch to get cv results")
        aggregated_results = evaluate(
            self,
            cv if cv else self.estimator.cv,
            y,
            X,
            scoring=scoring if scoring else self.estimator.scoring,
        ).reset_index(drop=True)
        return aggregated_results

    def _get_cv_results_clustered(self):
        if not isinstance(self.estimator, BaseGridSearch):
            raise ValueError("forecaster should be a BaseGridSearch to get cv results")
        clustered_results = (
            pd.concat(
                [
                    (
                        grid.cv_results_.assign(
                            **grid.cv_results_.params.apply(pd.Series), group=group
                        )
                    )
                    for group, grid in self.grids.items()
                ]
            )
            if self.grids
            else None
        ).reset_index(drop=True)
        return clustered_results
