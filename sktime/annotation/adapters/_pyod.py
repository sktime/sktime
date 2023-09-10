#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements outlier detection from pyOD."""

import numpy as np
from sklearn.base import clone

from sktime.annotation.base._base import BaseSeriesAnnotator
from sktime.utils.validation._dependencies import _check_soft_dependencies

__author__ = ["mloning", "satya-pattnaik", "fkiraly"]

import pandas as pd


class PyODAnnotator(BaseSeriesAnnotator):
    """Transformer that applies outlier detector from pyOD.

    Parameters
    ----------
    estimator : PyOD estimator
        See ``https://pyod.readthedocs.io/en/latest/`` documentation for a detailed
        description of all options.
    fmt : str {"dense", "sparse"}, optional (default="dense")
        Annotation output format:
        * If "sparse", a sub-series of labels for only the outliers in X is returned,
        * If "dense", a series of labels for all values in X is returned.
    labels : str {"indicator", "score"}, optional (default="indicator")
        Annotation output labels:
        * If "indicator", returned values are boolean, indicating whether a value is an
        outlier,
        * If "score", returned values are floats, giving the outlier score.
    """

    _tags = {"python_dependencies": "pyod"}

    def __init__(self, estimator, fmt="dense", labels="indicator"):
        self.estimator = estimator  # pyod estimator
        super().__init__(fmt=fmt, labels=labels)

    def _fit(self, X, Y=None):
        """Fit to training data.

        core logic

        Parameters
        ----------
        X : pd.DataFrame
            training data to fit model to, time series
        Y : pd.Series, optional
            ground truth annotations for training if annotator is supervised
        Returns
        -------
        self : returns a reference to self

        Notes
        -----
        Create fitted model that sets attributes ending in "_".
        """
        X_np = X.to_numpy()

        if len(X_np.shape) == 1:
            X_np = X_np.reshape(-1, 1)

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_np)

        return self

    def _predict(self, X):
        """Create annotations on test/deployment data.

        Parameters
        ----------
        X : pd.DataFrame - data to annotate, time series

        Returns
        -------
        Y : pd.Series - annotations for sequence X
            exact format depends on annotation type
        """
        fmt = self.fmt
        labels = self.labels

        X_np = X.to_numpy()

        if len(X_np.shape) == 1:
            X_np = X_np.reshape(-1, 1)

        Y_np = self.estimator_.predict(X_np)

        if labels == "score":
            Y_val_np = self.estimator_.decision_function(X_np)
        elif labels == "indicator":
            Y_val_np = Y_np

        if fmt == "dense":
            Y = pd.Series(Y_val_np, index=X.index)
        elif fmt == "sparse":
            Y_loc = np.where(Y_np)
            Y = pd.Series(Y_val_np[Y_loc], index=X.index[Y_loc])

        return Y

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        if _check_soft_dependencies("pyod", severity="none"):
            from pyod.models.knn import KNN

            params = {"estimator": KNN()}
        else:
            params = {"estimator": "foo"}
        return params
