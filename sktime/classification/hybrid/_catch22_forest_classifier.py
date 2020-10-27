# -*- coding: utf-8 -*-
""" catch22 Forest Classifier
A forest classifier based on catch22 features
"""

__author__ = ["Carl Lubba", "Matthew Middlehurst"]
__all__ = ["Catch22ForestClassifier"]

import sys

import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn import tree, preprocessing
from sklearn.utils import check_random_state
from sklearn.utils import check_array
from catch22 import catch22_all
from sktime.classification.base import BaseClassifier


class Catch22ForestClassifier(BaseClassifier):
    """ Canonical Time-series Characteristics (catch22)

     @article{lubba2019catch22,
          title={catch22: CAnonical Time-series CHaracteristics},
          author={Lubba, Carl H and Sethi, Sarab S and Knaute, Philip and
                    Schultz, Simon R and Fulcher, Ben D and Jones, Nick S},
          journal={Data Mining and Knowledge Discovery},
          volume={33},
          number={6},
          pages={1821--1852},
          year={2019},
          publisher={Springer}
     }

     Overview: Input n series length m
     Transforms series into the 22 catch22 features extracted from the hctsa
     toolbox and builds a random forest classifier on them.

     Fulcher, B. D., & Jones, N. S. (2017). hctsa: A computational framework
     for automated time-series phenotyping using massive feature extraction.
     Cell systems, 5(5), 527-531.

     Fulcher, B. D., Little, M. A., & Jones, N. S. (2013). Highly comparative
     time-series analysis: the empirical structure of time series and their
     methods. Journal of the Royal Society Interface, 10(83), 20130048.

     Original Catch22ForestClassifier:
     https://github.com/chlubba/sktime-catch22

     catch22 package implementations:
     https://github.com/chlubba/catch22

     For the Java version, see
     https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
     /tsml/transformers/Catch22.java


     Parameters
     ----------
     n_estimators            : int, number of trees in the random forest
     bootstrap               : bool, if true draw samples with replacement
     n_jobs                  : int or None, number of jobs to run in parallel
     random_state            : int or None, seed for random, integer,
     optional (default to no seed)

     Attributes
     ----------
     bagging_classifier      : trained forest classifier

     """
    def __init__(
        self,
        n_estimators=100,
        bootstrap=True,
        n_jobs=None,
        random_state=None
    ):
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.bagging_classifier_ = None
        self.n_timestep_ = 0
        self.n_dims_ = 0
        self.classes_ = []
        super(Catch22ForestClassifier, self).__init__()

    def fit(self, X, y, sample_weight=None, check_input=True):
        """Fit a random catch22 feature forest classifier

            Parameters
            ----------
            X : nested pandas DataFrame of shape [n_instances, 1]
                Nested dataframe with univariate time-series in cells.
            y : array-like, shape = [n_instances] The class labels.

            Returns
            -------
            self : object
        """
        if sys.platform == 'win32':
            # todo update when catch22 is fixed for windows/alternative is made
            raise OSError("Catch22 does not support Windows OS currently.")

        # Correct formating of x
        if len(X.iloc[0]) == 1:  # UNI
            X = [
                np.array(X.iloc[i].iloc[0]).tolist()
                for i in range(0, len(X))
            ]
        else:  # MULTI
            X = [
                [
                    np.array(X.iloc[i].iloc[j]).tolist()
                    for j in range(0, len(X.iloc[i]))
                ] for i in range(0, len(X))
            ]

        random_state = check_random_state(self.random_state)

        self.classes_ = np.unique(y)
        le = preprocessing.LabelEncoder()
        y = le.fit_transform(y)

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")
            y = check_array(y, ensure_2d=False)

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError("illegal input dimension")

        n_samples = X.shape[0]
        self.n_timestep_ = X.shape[-1]
        if X.ndim > 2:
            n_dims = X.shape[1]
        else:
            n_dims = 1

        self.n_dims_ = n_dims

        if y.ndim == 1:
            self.classes_, y = np.unique(y, return_inverse=True)
        else:
            _, y = np.nonzero(y)
            if len(y) != n_samples:
                raise ValueError("Single label per sample expected.")
            self.classes_ = np.unique(y)

        if len(y) != n_samples:
            raise ValueError("Number of labels={} does not match "
                             "number of samples={}".format(len(y), n_samples))

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        if not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=np.intp)

        clf = tree.DecisionTreeClassifier(
            class_weight="balanced",
            random_state=random_state
        )

        self.bagging_classifier_ = BaggingClassifier(
            base_estimator=clf,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )
        X = X.reshape(n_samples, n_dims * self.n_timestep_)

        # compute catch22 features
        num_insts = X.shape[0]
        X_catch22 = []
        for i in range(num_insts):
            series = X[i, :]
            c22_dict = catch22_all(series)
            X_catch22.append(c22_dict['values'])

        # replace the rare nans
        X_catch22 = np.array(X_catch22)
        X_catch22[np.logical_or(np.isnan(X_catch22), np.isinf(X_catch22))] = 0

        self.bagging_classifier_.fit(X_catch22, y, sample_weight=sample_weight)

        self._is_fitted = True
        return self

    def predict(self, X, check_input=True):
        return self.classes_[
            np.argmax(self.predict_proba(X, check_input=check_input), axis=1)
        ]

    def predict_proba(self, X, check_input=True):
        self.check_is_fitted()

        # Correct formating of x
        if len(X.iloc[0]) == 1:  # UNI
            X = [
                np.array(X.iloc[i].iloc[0]).tolist()
                for i in range(0, len(X))
            ]
        else:  # MULTI
            X = [
                [
                    np.array(X.iloc[i].iloc[j]).tolist()
                    for j in range(0, len(X.iloc[i]))
                ] for i in range(0, len(X))
            ]

        if check_input:
            X = check_array(X, dtype=np.float64, allow_nd=True, order="C")

        if X.ndim < 2 or X.ndim > 3:
            raise ValueError(
                "illegal input dimensions X.ndim ({})".format(X.ndim)
            )

        if self.n_dims_ > 1 and X.ndim != 3:
            raise ValueError("illegal input dimensions X.ndim != 3")

        if X.shape[-1] != self.n_timestep_:
            raise ValueError(
                "illegal input shape ({} != {})".format(
                    X.shape[-1],
                    self.n_timestep_
                )
            )

        if X.ndim > 2 and X.shape[1] != self.n_dims_:
            raise ValueError(
                "illegal input shape ({} != {}".format(
                    X.shape[1],
                    self.n_dims
                )
            )

        if X.dtype != np.float64 or not X.flags.contiguous:
            X = np.ascontiguousarray(X, dtype=np.float64)

        X = X.reshape(X.shape[0], self.n_dims_ * self.n_timestep_)

        # compute catch22 features
        num_insts = X.shape[0]
        X_catch22 = []
        for i in range(num_insts):
            series = X[i, :]
            c22_dict = catch22_all(series)
            X_catch22.append(c22_dict['values'])

        # replace the rare nans
        X_catch22 = np.array(X_catch22)
        X_catch22[np.logical_or(np.isnan(X_catch22), np.isinf(X_catch22))] = 0

        return self.bagging_classifier_.predict_proba(X_catch22)
