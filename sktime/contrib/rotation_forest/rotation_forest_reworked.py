# -*- coding: utf-8 -*-
__author__ = "Markus Löning"
__all__ = ["RotationForestClassifier"]

from warnings import warn

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.ensemble._forest import ForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y
from sklearn.utils.validation import check_array
from sklearn.utils.validation import check_random_state
from sktime.classification.base import BaseClassifier


class RotationForestClassifier(BaseClassifier, ForestClassifier):
    """Rotation Forest Classifier

    Parameters
    ----------
    n_estimators :
    n_column_subsets
    p_instance_subset
    random_state
    verbose

    References
    ----------
    @article{Rodriguez2006,
        author = {Juan J. Rodriguez and Ludmila I. Kuncheva and Carlos J.
        Alonso},
        journal = {IEEE Transactions on Pattern Analysis and Machine
        Intelligence},
        number = {10},
        pages = {1619-1630},
        title = {Rotation Forest: A new classifier ensemble method},
        volume = {28},
        year = {2006},
        ISSN = {0162-8828},
        URL = {http://doi.ieeecomputersociety.org/10.1109/TPAMI.2006.211}
    }
    @article{bagnall2018rotation,
      title={Is rotation forest the best classifier for problems with
      continuous features?},
      author={Bagnall, A and Bostrom, Aaron and Cawley, G and Flynn, Michael
      and Large, James and Lines, Jason},
      journal={arXiv preprint arXiv:1809.06705},
      year={2018}
    }

    Java reference implementation:
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java
    /weka/classifiers/meta/RotationForest.java
    """

    def __init__(
        self,
        n_estimators=200,
        n_column_subsets=3,
        p_instance_subset=0.75,
        random_state=None,
        verbose=0,
    ):

        super(RotationForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(), n_estimators=n_estimators
        )

        # settable parameters
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.p_instance_subset = p_instance_subset
        self.n_column_subsets = n_column_subsets

        # get random state object
        self._rng = check_random_state(self.random_state)

        # fixed parameters
        self.base_transformer = PCA(random_state=random_state)
        self.base_estimator = DecisionTreeClassifier(random_state=random_state)

        # defined in fit
        self.estimators_ = []
        self.column_subsets_ = {}
        self.transformers_ = {}
        self.n_columns_ = None
        self.classes_ = None
        self.n_outputs_ = None
        self.n_instances_ = None
        self.n_instances_in_subset = None

    def fit(self, X, y):
        # check inputs
        X, y = check_X_y(X, y)

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn(
                "A column-vector y was passed when a 1d array was"
                " expected. Please change the shape of y to "
                "(n_samples,), for example using ravel().",
                DataConversionWarning,
                stacklevel=2,
            )

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        self.n_instances_, self.n_columns_ = X.shape
        self.classes_ = np.unique(y)
        self.n_outputs_ = y.shape[1]

        # get number of instances in random subsets
        self.n_instances_in_subset = int(self.n_instances_ * self.p_instance_subset)

        # check if there are at least as many samples as columns in subset
        # for PCA,
        # as n_components will be min(n_samples, n_columns)
        n_columns_in_subset = int(np.ceil(self.n_columns_ / self.n_column_subsets))
        if self.n_instances_in_subset < n_columns_in_subset:
            raise ValueError(
                "There are fewer instances than columns in random subsets, "
                "hence PCA cannot compute components for all columns, please "
                "change `n_column_subsets` or `p_instance_subset`"
            )

        # Z-normalise the data
        X_norm = self._normalise_X(X)

        # preallocate matrix for transformed data
        Xt = np.zeros((self.n_instances_, self.n_columns_))

        # TODO: parallelize
        for i in range(self.n_estimators):

            # randomly split columns into disjoint subsets
            columns = np.arange(self.n_columns_)
            self._rng.shuffle(columns)
            self.column_subsets_[i] = np.array_split(columns, self.n_column_subsets)

            # initialise list of transformers
            self.transformers_[i] = []

            for column_subset in self.column_subsets_[i]:
                # select random subset of instances by classes
                instance_subset = self._get_random_instance_subset_by_classes(y)

                # fit transformer on subset of instances and columns
                transformer = clone(self.base_transformer)
                transformer.fit(X_norm[instance_subset, column_subset])
                self.transformers_[i].append(transformer)

                # transform on subset of columns but all instances
                Xt[:, column_subset] = transformer.transform(X_norm[:, column_subset])

            # fit estimator on transformed data
            estimator = clone(self.base_estimator)
            estimator.fit(Xt, y)
            self.estimators_.append(estimator)

        self._is_fitted = True
        return self

    def _get_random_instance_subset_by_classes(self, y):
        """Helper function to select bootstrap subset of instances for given
        random subset of classes"""
        # get random state object
        rng = self._rng

        # get random subset by class
        n_classes = rng.randint(1, len(self.classes_) + 1)
        classes = rng.choice(self.classes_, size=n_classes, replace=False)

        # get instances for selected classes
        isin_classes = np.where(np.isin(y, classes))[0]

        # randomly select bootstrap subset of instances for selected classes
        instance_subset = rng.choice(
            isin_classes, size=self.n_instances_in_subset, replace=True
        )
        return instance_subset[:, None]

    def _normalise_X(self, X):
        """Helper function to normalise X using the z-score standardisation"""
        # Xt = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        scaler = StandardScaler(with_mean=True, with_std=True)
        Xt = scaler.fit_transform(X)
        return Xt

    def predict_proba(self, X):
        """Predict probabilities"""
        self.check_is_fitted()

        # check input
        X = check_array(X)

        # normalise data
        X_norm = self._normalise_X(X)

        # TODO parallelize
        all_proba = []
        for i, estimator in enumerate(self.estimators_):

            # transform data using fitted transformers
            Xt = np.zeros(X_norm.shape)
            for j, column_subset in enumerate(self.column_subsets_[i]):
                # get fitted transformer
                transformer = self.transformers_[i][j]

                # transform data
                Xt[:, column_subset] = transformer.transform(X_norm[:, column_subset])

            # predict on transformed data
            proba = estimator.predict_proba(Xt)
            all_proba.append(proba)

        # aggregate predicted probabilities
        all_proba = np.sum(all_proba, axis=0) / len(self.estimators_)

        return all_proba
