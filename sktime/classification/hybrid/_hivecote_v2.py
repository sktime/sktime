# -*- coding: utf-8 -*-
"""Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2.

Upgraded hybrid ensemble of classifiers from 4 separate time series classification
representations, using the weighted probabilistic CAWPE as an ensemble controller.
"""

__author__ = "Matthew Middlehurst"
__all__ = ["HIVECOTEV2"]

from datetime import datetime

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.classification.dictionary_based import TemporalDictionaryEnsemble
from sktime.classification.interval_based._drcif import DrCIF
from sktime.classification.kernel_based import Arsenal
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.utils.validation import check_n_jobs


class HIVECOTEV2(BaseClassifier):
    """Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V2.

    An ensemble of the STC, DrCIF, Arsenal and TDE classifiers from different feature
    representations using the CAWPE structure as described in [1].

    Parameters
    ----------
    stc_params : dict or None, default=None
        Parameters for the ShapeletTransformClassifier module. If None, uses the
        default parameters.
    drcif_params : dict or None, default=None
        Parameters for the DrCIF module. If None, uses the default parameters with
        n_estimators set to 500.
    arsenal_params : dict or None, default=None
        Parameters for the Arsenal module. If None, uses the default parameters.
    tde_params : dict or None, default=None
        Parameters for the TemporalDictionaryEnsemble module. If None, uses the default
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
    HIVECOTEV1, ShapeletTransformClassifier, DrCIF, Arsenal, TemporalDictionaryEnsemble

    Notes
    -----
    For the Java version, see
    `https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/hybrids/HIVE_COTE.java`_.

    References
    ----------
    .. [1] Middlehurst, Matthew, James Large, Michael Flynn, Jason Lines, Aaron Bostrom,
       and Anthony Bagnall. "HIVE-COTE 2.0: a new meta ensemble for time series
       classification." arXiv preprint arXiv:2104.07551 (2021).

    Examples
    --------
    >>> from sktime.classification.hybrid import HIVECOTEV2
    >>> from sktime.contrib.vector_classifiers._rotation_forest import RotationForest
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train", return_X_y=True)
    >>> X_test, y_test = load_unit_test(split="test", return_X_y=True)
    >>> clf = HIVECOTEV2(
    ...     stc_params={
    ...         "estimator": RotationForest(n_estimators=10),
    ...         "n_shapelets_considered": 500,
    ...         "max_shapelets": 10,
    ...         "batch_size": 30,
    ...     },
    ...     drcif_params={"n_estimators": 10},
    ...     arsenal_params={"num_kernels": 100, "n_estimators": 5},
    ...     tde_params={
    ...         "n_parameter_samples": 25,
    ...         "max_ensemble_size": 5,
    ...         "randomly_selected_params": 10,
    ...     },
    ... )
    >>> clf.fit(X_train, y_train)
    HIVECOTEV2(...)
    >>> y_pred = clf.predict(X_test)
    """

    _tags = {
        "capability:multivariate": True,
        "capability:unequal_length": False,
        "capability:missing_values": False,
        "capability:train_estimate": True,
        "capability:contractable": True,
    }

    def __init__(
        self,
        stc_params=None,
        drcif_params=None,
        arsenal_params=None,
        tde_params=None,
        time_limit_in_minutes=0,
        verbose=0,
        n_jobs=1,
        random_state=None,
    ):
        self.stc_params = stc_params
        self.drcif_params = drcif_params
        self.arsenal_params = arsenal_params
        self.tde_params = tde_params

        self.time_limit_in_minutes = time_limit_in_minutes

        self.verbose = verbose
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.n_classes = 0
        self.classes_ = []

        self._stc_params = stc_params
        self._drcif_params = drcif_params
        self._arsenal_params = arsenal_params
        self._tde_params = tde_params
        self._n_jobs = n_jobs
        self._stc = None
        self._drcif = None
        self._arsenal = None
        self._tde = None
        self._stc_weight = 0
        self._drcif_weight = 0
        self._arsenal_weight = 0
        self._tde_weight = 0

        super(HIVECOTEV2, self).__init__()

    def _fit(self, X, y):
        self._n_jobs = check_n_jobs(self.n_jobs)

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        if self.stc_params is None:
            self._stc_params = {"transform_limit_in_minutes": 120}
        if self.drcif_params is None:
            self._drcif_params = {"n_estimators": 500}
        if self.arsenal_params is None:
            self._arsenal_params = {}
        if self.tde_params is None:
            self._tde_params = {}

        if self.time_limit_in_minutes > 0:
            # leave 1/3 for train estimate
            ct = self.time_limit_in_minutes / 6
            self._stc_params["time_limit_in_minutes"] = ct
            self._drcif_params["time_limit_in_minutes"] = ct
            self._arsenal_params["time_limit_in_minutes"] = ct
            self._tde_params["time_limit_in_minutes"] = ct

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

        self._drcif = DrCIF(
            **self._drcif_params,
            save_transformed_data=True,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        self._drcif.fit(X, y)

        if self.verbose > 0:
            print("DrCIF ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_probs = self._drcif._get_train_probs(X, y)
        train_preds = self._drcif.classes_[np.argmax(train_probs, axis=1)]
        self._drcif_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "DrCIF train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("DrCIF weight = " + str(self._drcif_weight))  # noqa

        self._arsenal = Arsenal(
            **self._arsenal_params,
            save_transformed_data=True,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        self._arsenal.fit(X, y)

        if self.verbose > 0:
            print("Arsenal ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_probs = self._arsenal._get_train_probs(X, y)
        train_preds = self._arsenal.classes_[np.argmax(train_probs, axis=1)]
        self._arsenal_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "Arsenal train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("Arsenal weight = " + str(self._arsenal_weight))  # noqa

        self._tde = TemporalDictionaryEnsemble(
            **self._tde_params,
            save_train_predictions=True,
            random_state=self.random_state,
            n_jobs=self._n_jobs,
        )
        self._tde.fit(X, y)

        if self.verbose > 0:
            print("TDE ", datetime.now().strftime("%H:%M:%S %d/%m/%Y"))  # noqa

        train_probs = self._tde._get_train_probs(X, y, train_estimate_method="oob")
        train_preds = self._tde.classes_[np.argmax(train_probs, axis=1)]
        self._tde_weight = accuracy_score(y, train_preds) ** 4

        if self.verbose > 0:
            print(  # noqa
                "TDE train estimate ",
                datetime.now().strftime("%H:%M:%S %d/%m/%Y"),
            )
            print("TDE weight = " + str(self._tde_weight))  # noqa

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
            self._drcif.predict_proba(X)
            * (np.ones(self.n_classes) * self._drcif_weight),
        )
        dists = np.add(
            dists,
            self._arsenal.predict_proba(X)
            * (np.ones(self.n_classes) * self._arsenal_weight),
        )
        dists = np.add(
            dists,
            self._tde.predict_proba(X) * (np.ones(self.n_classes) * self._tde_weight),
        )

        return dists / dists.sum(axis=1, keepdims=True)
