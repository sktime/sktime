# -*- coding: utf-8 -*-
"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V1.

Hybrid ensemble of classifiers from 4 separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
"""

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
from sktime.utils.validation import check_n_jobs


class HIVECOTEV1(BaseClassifier):
    """Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V1.

    An ensemble of the STC, TSF, RISE and cBOSS classifiers from different feature
    representations using the CAWPE structure as described in [1].

    Parameters
    ----------
    stc_params : dict or None, default=None
        Parameters for the ShapeletTransformClassifier module. If None, uses the
        default parameters.
    tsf_params : dict or None, default=None
        Parameters for the TimeSeriesForestClassifier module. If None, uses the default
        parameters with n_estimators set to 500.
    rise_params : dict or None, default=None
        Parameters for the RandomIntervalSpectralForest module. If None, uses the
        default parameters with n_estimators set to 500.
    cboss_params : dict or None, default=None
        Parameters for the ContractableBOSS module. If None, uses the default
        parameters.
    verbose : int, default=0
        Level of output printed to the console (for information only).
    n_jobs : int, default=1
        The number of jobs to run in parallel for both `fit` and `predict`.
        ``-1`` means using all processors.
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    n_classes : int
        The number of classes.
    classes_ : list
        The classes labels.

    See Also
    --------
    HIVECOTEV2

    Notes
    -----
    For the Java version, see
    `https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/hybrids/HIVE_COTE.java`_.

    References
    ----------
    .. [1] Anthony Bagnall, Michael Flynn, James Large, Jason Lines and
       Matthew Middlehurst. "On the usage and performance of the Hierarchical Vote
       Collective of Transformation-based Ensembles version 1.0 (hive-cote v1.0)"
       International Workshop on Advanced Analytics and Learning on Temporal Data 2020

    Examples
    --------
    >>> from sktime.classification.hybrid import HIVECOTEV1
    >>> from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> rotf = RotationForest(n_estimators=3)
    >>> clf = HIVECOTEV1(
    ...     stc_params={"estimator": rotf, "transform_limit_in_minutes": 0.025},
    ...     tsf_params={"n_estimators": 3},
    ...     rise_params={"n_estimators": 3},
    ...     cboss_params={"n_parameter_samples": 10, "max_ensemble_size": 3},
    ... )
    >>> clf.fit(X_train, y_train)
    HIVECOTEV1(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": False,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": False,
        "capability:contractable": False,
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
        self.stc_params = stc_params
        self.tsf_params = tsf_params
        self.rise_params = rise_params
        self.cboss_params = cboss_params

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.n_classes = 0
        self.classes_ = []

        self._stc_params = stc_params
        self._tsf_params = tsf_params
        self._rise_params = rise_params
        self._cboss_params = cboss_params
        self._n_jobs = n_jobs
        self._stc = None
        self._tsf = None
        self._rise = None
        self._cboss = None
        self._stc_weight = 0
        self._tsf_weight = 0
        self._rise_weight = 0
        self._cboss_weight = 0

        super(HIVECOTEV1, self).__init__()

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        if self.stc_params is None:
            self._stc_params = {}
        if self.tsf_params is None:
            self._tsf_params = {"n_estimators": 500}
        if self.rise_params is None:
            self._rise_params = {"n_estimators": 500}
        if self.cboss_params is None:
            self._cboss_params = {}

        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < cv_size:
            cv_size = min_class

        self._stc = ShapeletTransformClassifier(
            **self._stc_params,
            save_transformed_data=True,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        self._stc.fit(X, y)

        if self.verbose > 0:
            print("STC ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_probs = self._stc._get_train_probs(X, y)
        train_preds = self._stc.classes_[np.argmax(train_probs, axis=1)]
        self._stc_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "STC train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("STC weight = " + str(self._stc_weight))  # noqa

        self._tsf = TimeSeriesForestClassifier(
            **self._tsf_params,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        self._tsf.fit(X, y)

        if self.verbose > 0:
            print("TSF ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_preds = cross_val_predict(
            TimeSeriesForestClassifier(
                **self._tsf_params, random_state=self.random_state
            ),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self._n_jobs,
        )
        self._tsf_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "TSF train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("TSF weight = " + str(self._tsf_weight))  # noqa

        self._rise = RandomIntervalSpectralForest(
            **self._rise_params,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        self._rise.fit(X, y)

        if self.verbose > 0:
            print("RISE ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_preds = cross_val_predict(
            RandomIntervalSpectralForest(
                **self._rise_params,
                random_state=self.random_state,
            ),
            X=X,
            y=y,
            cv=cv_size,
            n_jobs=self._n_jobs,
        )
        self._rise_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "RISE train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("RISE weight = " + str(self._rise_weight))  # noqa

        self._cboss = ContractableBOSS(
            **self._cboss_params, random_state=self.random_state, n_jobs=self._n_jobs
        )
        self._cboss.fit(X, y)
        train_probs = self._cboss._get_train_probs(X, y)
        train_preds = self._cboss.classes_[np.argmax(train_probs, axis=1)]
        self._cboss_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "cBOSS (estimate included)",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("cBOSS weight = " + str(self._cboss_weight))  # noqa

    def _predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def _predict_proba(self, X):
        dists = np.zeros((X.shape[0], self.n_classes))

        dists = np.add(
            dists,
            self._stc.predict_proba(X) * (np.ones(self.n_classes) * self._stc_weight),
        )
        dists = np.add(
            dists,
            self._tsf.predict_proba(X) * (np.ones(self.n_classes) * self._tsf_weight),
        )
        dists = np.add(
            dists,
            self._rise.predict_proba(X) * (np.ones(self.n_classes) * self._rise_weight),
        )
        dists = np.add(
            dists,
            self._cboss.predict_proba(X)
            * (np.ones(self.n_classes) * self._cboss_weight),
        )

        return dists / dists.sum(axis=1, keepdims=True)
