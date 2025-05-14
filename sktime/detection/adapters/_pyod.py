#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements outlier detection from pyOD."""

import numpy as np
from sklearn.base import clone

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _check_soft_dependencies

__author__ = ["mloning", "satya-pattnaik", "fkiraly"]

import pandas as pd


class PyODDetector(BaseDetector):
    """Transformer that applies outlier detector from pyOD.

    Parameters
    ----------
    estimator : PyOD estimator
        See ``https://pyod.readthedocs.io/en/latest/`` documentation for a detailed
        description of all options.

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.detection.datagen import piecewise_normal_multivariate
    >>> X = pd.DataFrame(piecewise_normal_multivariate(
    ...     means=[[1, 3], [4, 5]],
    ...     lengths=[3, 4],
    ...     random_state=10),
    ... )
    >>> from sktime.detection.adapters._pyod import PyODDetector
    >>> from pyod.models.ecod import ECOD
    >>> model = PyODDetector(ECOD())
    >>> model.fit_transform(X)
       labels
    0       0
    1       1
    2       0
    3       0
    4       0
    5       0
    6       0
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["mloning", "satya-pattnaik", "fkiraly"],
        "maintainers": "satya-pattnaik",
        # estimator type
        # --------------
        "python_dependencies": "pyod",
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
    }

    def __init__(self, estimator, labels="indicator"):
        self.estimator = estimator  # pyod estimator
        self.labels = labels

        super().__init__()

    def _fit(self, X, y=None):
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
        X : pd.DataFrame
            Time series subject to detection, which will be assigned labels or scores.

        Returns
        -------
        Y : pd.Series - annotations for sequence X
        """
        labels = self.labels

        X_np = X.to_numpy()

        if len(X_np.shape) == 1:
            X_np = X_np.reshape(-1, 1)

        Y_np = self.estimator_.predict(X_np)

        if labels == "score":
            Y_val_np = self.estimator_.decision_function(X_np)
        elif labels == "indicator":
            Y_val_np = Y_np

        Y_loc = np.where(Y_np)
        Y = pd.Series(Y_val_np[Y_loc])

        return Y

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            There are currently no reserved values for annotators.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        if _check_soft_dependencies("pyod", severity="none"):
            from pyod.models.knn import KNN

            params0 = {"estimator": KNN()}
            params1 = {"estimator": KNN(n_neighbors=3)}
        else:
            params0 = {"estimator": "foo"}
            params1 = {"estimator": "bar"}
        return [params0, params1]


# todo 1.0.0 - remove alias, i.e., remove this line
PyODAnnotator = PyODDetector
