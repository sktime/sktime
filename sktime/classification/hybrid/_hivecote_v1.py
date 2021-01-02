# -*- coding: utf-8 -*-
""" Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V1
"""

__author__ = "Matthew Middlehurst"
__all__ = ["HIVECOTEV1"]

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import class_distribution

from sktime.classification.base import BaseClassifier
from sktime.classification.dictionary_based import ContractableBOSS
from sktime.classification.interval_based import (
    TimeSeriesForest,
    RandomIntervalSpectralForest,
)
from sktime.classification.shapelet_based import ShapeletTransformClassifier
from sktime.utils.validation.panel import check_X_y, check_X


class HIVECOTEV1(BaseClassifier):
    """
    Hierarchical Vote Collective of Transformation-based Ensembles (HIVE-COTE) V1
    as described in [1].

    An ensemble of the STC, TSF, RISE and cBOSS classifiers from different feature
    representations using the CAWPE structure.


    Parameters
    ----------
    random_state            : int or None, seed for random, integer,
    optional (default to no seed)

    Attributes
    ----------
    n_classes               : extracted from the data

    Notes
    -----
    @article{bagnall2020usage,
      title={On the Usage and Performance of The Hierarchical Vote Collective of
      Transformation-based Ensembles version 1.0 (HIVE-COTE 1.0)},
      author={Bagnall, Anthony and Flynn, Michael and Large, James and Lines, Jason and
      Middlehurst, Matthew},
      journal={arXiv preprint arXiv:2004.06069},
      year={2020}
    }

    Java version
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/
    tsml/classifiers/hybrids/HIVE_COTE.java

    """

    # Capability tags
    capabilities = {
        "multivariate": False,
        "unequal_length": False,
        "missing_values": False,
    }

    def __init__(
        self,
        random_state=None,
    ):
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
        X, y = check_X_y(X, y, enforce_univariate=True, coerce_to_numpy=True)

        self.n_classes = np.unique(y).shape[0]
        self.classes_ = class_distribution(np.asarray(y).reshape(-1, 1))[0][0]

        cv_size = 10
        _, counts = np.unique(y, return_counts=True)
        min_class = np.min(counts)
        if min_class < cv_size:
            cv_size = min_class

        self.stc = ShapeletTransformClassifier(
            random_state=self.random_state,
            time_contract_in_mins=60,
        )
        self.stc.fit(X, y)
        train_preds = cross_val_predict(
            ShapeletTransformClassifier(
                random_state=self.random_state,
                time_contract_in_mins=60,
            ),
            X=X,
            y=y,
            cv=cv_size,
        )
        self.stc_weight = accuracy_score(y, train_preds) ** 4

        self.tsf = TimeSeriesForest(random_state=self.random_state)
        self.tsf.fit(X, y)
        train_preds = cross_val_predict(
            TimeSeriesForest(random_state=self.random_state),
            X=X,
            y=y,
            cv=cv_size,
        )
        self.tsf_weight = accuracy_score(y, train_preds) ** 4

        self.rise = RandomIntervalSpectralForest(random_state=self.random_state)
        self.fit(X, y)
        train_preds = cross_val_predict(
            RandomIntervalSpectralForest(random_state=self.random_state),
            X=X,
            y=y,
            cv=cv_size,
        )
        self.rise_weight = accuracy_score(y, train_preds) ** 4

        self.cboss = ContractableBOSS(random_state=self.random_state)
        self.cboss.fit(X, y)
        train_probs = self.cboss._get_train_probs(X)
        train_preds = self.cboss.classes_[np.argmax(train_probs, axis=1)]
        self.cboss_weight = accuracy_score(y, train_preds) ** 4

        return self

    def predict(self, X):
        rng = check_random_state(self.random_state)
        return np.array(
            [
                self.classes_[int(rng.choice(np.flatnonzero(prob == prob.max())))]
                for prob in self.predict_proba(X)
            ]
        )

    def predict_proba(self, X):
        self.check_is_fitted()
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)

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
