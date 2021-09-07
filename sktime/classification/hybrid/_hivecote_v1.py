# -*- coding: utf-8 -*-
"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V1."""

__author__ = "Matthew Middlehurst"
__all__ = ["HIVECOTEV1"]

from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.interval_based import (
    TimeSeriesForestClassifier,
    RandomIntervalSpectralForest,
)
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.utils.validation.panel import check_X_y, check_X


class HIVECOTEV1(BaseClassifier):
    """Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V1.

    An ensemble of the STC, TSF, RISE and cBOSS classifiers from different feature
    representations using the CAWPE structure as described in [1].

    Parameters
    ----------
    verbose                 : int, level of output printed to
    the console (for information only) (default = 0)
    n_jobs                  : int, optional (default=1)
    The number of jobs to run in parallel for both `fit` and `predict`.
    ``-1`` means using all processors.
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    n_classes               : extracted from the data

    Notes
    -----
    ..[1] Anthony Bagnall, Michael Flynn, James Large, Jason Lines and
    Matthew Middlehurst.
        "On the usage and performance of the Hierarchical Vote Collective of
            Transformation-based Ensembles version 1.0 (hive-cote v1. 0)"
        International Workshop on Advanced Analytics and Learning on Temporal
            Data 2020

    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/hybrids/HIVE_COTE.java

    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
        "train_estimate": False,
        "contractable": False,
    }

    def __init__(
        self,
        stc_params=None,
        tsf_params=None,
        rise_params=None,
        cboss_params=None,
        verbose=0,
        n_jobs=1,
        random_state=None,
    ):
        if stc_params is None:
            stc_params = {}
        if tsf_params is None:
            tsf_params = {"n_estimators": 500}
        if rise_params is None:
            rise_params = {"n_estimators": 500}
        if cboss_params is None:
            cboss_params = {}

        self.stc_params = stc_params
        self.tsf_params = tsf_params
        self.rise_params = rise_params
        self.cboss_params = cboss_params

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.stc = None
        self.tsf = None
        self.rise = None
        self.cboss = None

        self.stc_weight = 0
        self.tsf_weight = 0
        self.rise_weight = 0
        self.cboss_weight = 0

        self.n_classes = 0
        self.classes_ = []

        super(HIVECOTEV1, self).__init__()

    def fit(self, X, y):
        """Fit a HIVE-COTEv1.0 classifier.

        Parameters
        ----------
        X : nested pandas DataFrame of shape [n_instances, 1]
            Nested dataframe with univariate time-series in cells.
        y : array-like, shape = [n_instances] The class labels.

        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y, enforce_univariate=True)

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < cv_size:
            cv_size = min_class

        self.stc = ShapeletTransformClassifier(
            **self.stc_params,
            random_state=self.random_state,
        )
        self.stc.fit(X, y)

        if self.verbose > 0:
            print("STC ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_preds = cross_val_predict(
            ShapeletTransformClassifier(
                **self.stc_params,
                random_state=self.random_state,
            ),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self.n_jobs,
        )
        self.stc_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "STC train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("STC weight = " + str(self.stc_weight))  # noqa

        self.tsf = TimeSeriesForestClassifier(
            **self.tsf_params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.tsf.fit(X, y)

        if self.verbose > 0:
            print("TSF ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_preds = cross_val_predict(
            TimeSeriesForestClassifier(
                **self.tsf_params, random_state=self.random_state
            ),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self.n_jobs,
        )
        self.tsf_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "TSF train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("TSF weight = " + str(self.tsf_weight))  # noqa

        self.rise = RandomIntervalSpectralForest(
            **self.rise_params,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.rise.fit(X, y)

        if self.verbose > 0:
            print("RISE ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_preds = cross_val_predict(
            RandomIntervalSpectralForest(
                **self.rise_params,
                random_state=self.random_state,
            ),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self.n_jobs,
        )
        self.rise_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "RISE train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("RISE weight = " + str(self.rise_weight))  # noqa

        self.cboss = ContractableBOSS(
            **self.cboss_params, random_state=self.random_state, n_jobs=self.n_jobs
        )
        self.cboss.fit(X, y)
        train_probs = self.cboss._get_train_probs(X)
        train_preds = self.cboss.classes_[np.argmax(train_probs, axis=1)]
        self.cboss_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "cBOSS (estimate included) ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("cBOSS weight = " + str(self.cboss_weight))  # noqa

        self._is_fitted = True
        return self

    def predict(self, X):
        """Make predictions for all cases in X.

        Parameters
        ----------
        X : The testing input samples of shape [n_instances,1].

        Returns
        -------
        output : numpy array of shape = [n_instances]
        """
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        """Make class probability estimates on each case in X.

        Parameters
        ----------
        X - pandas dataframe of testing data of shape [n_instances,1].

        Returns
        -------
        output : numpy array of shape =
                [n_instances, num_classes] of probabilities
        """
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True)

        dists = np.zeros((X.shape[0], self.n_classes))

        dists = np.add(
            dists,
            self.stc.predict_proba(X) * (np.ones(self.n_classes) * self.stc_weight),
        )
        dists = np.add(
            dists,
            self.tsf.predict_proba(X) * (np.ones(self.n_classes) * self.tsf_weight),
        )
        dists = np.add(
            dists,
            self.rise.predict_proba(X) * (np.ones(self.n_classes) * self.rise_weight),
        )
        dists = np.add(
            dists,
            self.cboss.predict_proba(X) * (np.ones(self.n_classes) * self.cboss_weight),
        )

        return dists / dists.sum(axis=1, keepdims=True)
