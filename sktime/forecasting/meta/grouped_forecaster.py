#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
"""Implements model that fits different models for different clusters of timeseries."""

__author__ = ["boukepostma"]
__all__ = ["GroupedForecaster"]

import pandas as pd
from sklearn import clone

from sktime.forecasting.base import BaseForecaster


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

    :param estimator: the model/pipeline to be applied per group
    :param groups: the column of the dataframe to select as a grouping parameter set
    :param aggregations: dictionary mapping col names to aggregation methods/functions
    :param use_global_model: whether or not to fall back to a general model in case the
                            group parameter is not found during `.predict()`
    """

    def __init__(
        self,
        estimator,
        groups,
        aggregations,
        use_global_model=True,
    ):
        self.estimator = estimator
        self.groups = groups
        self.aggregations = aggregations
        self.use_global_model = use_global_model
        super(GroupedForecaster, self).__init__()

    def __aggregate_X(self, X, grouper):
        return X.groupby(grouper).agg(
            {key: val for key, val in self.aggregations.items() if key != "y"}
        )

    def __aggregate_y(self, y, grouper):
        return y.groupby(grouper).agg(self.aggregations["y"])

    def __fit_single_group(self, group, y, X=None, fh=None):
        try:
            return clone(self.estimator).fit(y, X, fh)
        except Exception as e:
            raise type(e)(f"Exception for group {group}: {e}")

    def __fit_grouped_estimator(self, y, X, fh=None):
        # Aggregate data on group-level to avoid multiple observations per timepoint
        grouper = X[self.groups]
        y = self.__aggregate_y(y, grouper)
        X = self.__aggregate_X(X, grouper)

        # Use these indices for every group
        group_indices = X.groupby(self.groups).indices

        grouped_estimators = {
            # Fit a clone of the transformer to each group
            group: self.__fit_single_group(group, y[indices], X[indices, :], fh)
            for group, indices in group_indices.items()
        }

        return grouped_estimators

    def _fit(self, y, X, fh=None):
        """
        Fit the model using X, y as training data.

        :param X: pd.DataFrame, shape=(n_columns, n_samples,)
                grouping columns and training data.
        :param y: pd.Series, shape=(n_samples,) training data.
        :return: Returns an instance of self.
        """
        self.fallback_ = None

        if self.use_global_model:
            self.fallback_ = clone(self.estimator).fit(y, X, fh)

        self.estimators_ = self.__fit_grouped_estimator(y, X, fh)

        self.groups_ = as_list(self.estimators_.keys())

        return self

    def __predict_single_group(self, group, fh=None, X=None):
        """Predict a single group by getting its estimator from the fitted dict."""
        try:
            group_predictor = self.estimators_[group]
        except KeyError:
            if self.fallback_:
                group_predictor = self.fallback_
            else:
                raise ValueError(
                    f"Found new group {group} during predict with"
                    "use_global_model = False"
                )

        return pd.DataFrame(group_predictor.predict(X))

    def _predict(self, fh=None, X=None):
        """
        Predict for all groups on new data.

        :param fh: XXX
        :param X: XXX
        :return: array, shape=(n_samples,) the predicted data
        """
        X = self.__aggregate_X(X, X[self.groups])
        if X:
            group_indices = X.groupby(self.groups).indices
            return (
                pd.concat(
                    [
                        self.__predict_single_group(group, fh, X.loc[indices, :])
                        for group, indices in group_indices.items()
                    ],
                    axis=0,
                )
                .sort_index()
                .values.squeeze()
            )
