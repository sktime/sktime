__author__ = ["Markus Löning"]
__all__ = ["RotationForestClassifier"]

from itertools import islice
from warnings import warn

import numpy as np
from sklearn.base import clone
from sklearn.decomposition import PCA
from sklearn.exceptions import DataConversionWarning
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import Normalizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_X_y, check_array, check_random_state, check_is_fitted

from sktime.classifiers.base import BaseClassifier


class RotationForestClassifier(BaseClassifier):
    """Rotation Forest Classifier

    References
    ----------
    @article{Rodriguez2006,
        author = {Juan J. Rodriguez and Ludmila I. Kuncheva and Carlos J. Alonso},
        journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence},
        number = {10},
        pages = {1619-1630},
        title = {Rotation Forest: A new classifier ensemble method},
        volume = {28},
        year = {2006},
        ISSN = {0162-8828},
        URL = {http://doi.ieeecomputersociety.org/10.1109/TPAMI.2006.211}
    }
    @article{bagnall2018rotation,
      title={Is rotation forest the best classifier for problems with continuous features?},
      author={Bagnall, A and Bostrom, Aaron and Cawley, G and Flynn, Michael and Large, James and Lines, Jason},
      journal={arXiv preprint arXiv:1809.06705},
      year={2018}
    }

    Java reference implementation:
    https://github.com/uea-machine-learning/tsml/blob/master/src/main/java/weka/classifiers/meta/RotationForest.java
    """

    def __init__(self,
                 n_estimators=200,
                 min_columns_subset=3,
                 max_columns_subset=3,
                 p_instance_subset=0.75,
                 bootstrap_instance_subset=False,
                 random_state=None,
                 verbose=True):

        # settable parameters
        self.verbose = verbose
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.p_instance_subset = p_instance_subset
        self.min_columns_subset = min_columns_subset
        self.max_columns_subset = max_columns_subset
        self.bootstrap_instance_subset = bootstrap_instance_subset

        # get random state object
        self._rng = check_random_state(self.random_state)

        # note that certain checks in fit depend on the transformer being pca,
        # the estimator should be more easily substitutable
        self.transformer = PCA(random_state=random_state)
        self.estimator = DecisionTreeClassifier(random_state=random_state)

        # defined in fit
        self.estimators_ = []
        self.transformers_ = {}
        self.classes_ = None
        self.n_outputs_ = None
        self.n_instances_ = None
        self.n_columns_ = None
        self.column_subsets_ = {}
        self.n_instances_in_subset_ = None

    def fit(self, X, y, **fit_kwargs):
        # check inputs
        X, y = check_X_y(X, y)

        y = np.atleast_1d(y)
        if y.ndim == 2 and y.shape[1] == 1:
            warn("A column-vector y was passed when a 1d array was"
                 " expected. Please change the shape of y to "
                 "(n_samples,), for example using ravel().",
                 DataConversionWarning, stacklevel=2)

        if y.ndim == 1:
            # reshape is necessary to preserve the data contiguity against vs
            # [:, np.newaxis] that does not.
            y = np.reshape(y, (-1, 1))

        # get input shapes
        self.n_instances_, self.n_columns_ = X.shape
        self.classes_ = np.unique(y)
        self.n_outputs_ = y.shape[1]

        # compute number of instances and columns to be considered in each random subset
        self.n_instances_in_subset_ = int(self.n_instances_ * self.p_instance_subset)

        # preprocess the data
        X_norm = self._preprocess_X(X)

        # preallocate matrix for transformed data
        Xt = np.zeros((self.n_instances_, self.n_columns_))

        # TODO: parallelize
        for i in range(self.n_estimators):

            # randomly split columns into disjoint subsets
            columns = np.arange(self.n_columns_)
            self.column_subsets_[i] = self._random_column_subsets(columns, min_length=self.min_columns_subset,
                                                                  max_length=self.max_columns_subset)

            # initialise list of transformers
            self.transformers_[i] = []

            for column_subset in self.column_subsets_[i]:
                # select random subset of instances by classes
                classes, instance_subset = self._random_instance_subset(y, n_instances=self.n_instances_in_subset_,
                                                                        bootstrap=self.bootstrap_instance_subset)

                # check if there are at least as many samples as columns in subset for PCA,
                # as n_components will be min(n_samples, n_columns), otherwise throws error
                # when assigning transformed data
                if len(instance_subset) < len(column_subset):
                    if self.verbose:
                        warn("There are fewer instances than columns in random subsets, "
                             "hence PCA cannot compute components for all columns, randomly added"
                             "more bootstrapped instances. To avoid this, please "
                             "reduce `max_columns_subset` or increase `p_instance_subset`")
                    _, new_instance_subset = self._random_instance_subset(y, n_instances=10, classes=classes,
                                                                          bootstrap=True)
                    instance_subset = np.vstack([instance_subset, new_instance_subset])

                # try to fit transformer on subset of instances and columns, if it fails, add more instances and try
                # again
                n_attempts = 0  # keep track of number of time it fails
                while True:
                    transformer = clone(self.transformer)

                    # ignore error state on PCA because we account for it if it fails
                    with np.errstate(divide='ignore', invalid='ignore'):
                        transformer.fit(X_norm[instance_subset, column_subset])

                    # if fitting pca failed, add random samples and try fitting again
                    if np.any(np.isnan(transformer.explained_variance_ratio_)) or np.any(np.isinf(
                            transformer.explained_variance_ratio_)):
                        if self.verbose:
                            warn("PCA failed to fit on subset, randomly adding more bootstrapped "
                                 "instances and try to refit")
                        n_attempts += 1
                        _, new_instance_subset = self._random_instance_subset(y, n_instances=10, classes=classes,
                                                                              bootstrap=True)
                        instance_subset = np.vstack([instance_subset, new_instance_subset])

                        # raise error after 10 failed tries
                        if n_attempts == 10:
                            raise ValueError(f"Cannot fit PCA on subset, repeatedly added more random instances "
                                             f"to subset but keeps failing")

                    # otherwise break while-loop and continue
                    else:
                        break

                # append fitted transformer
                self.transformers_[i].append(transformer)

                # transform on subset of columns but all instances
                Xt[:, column_subset] = transformer.transform(X_norm[:, column_subset])

            # fit estimator on transformed data
            estimator = clone(self.estimator)
            estimator.fit(Xt, y)

            # append fitted estimator
            self.estimators_.append(estimator)

        self._is_fitted = True
        return self

    def _random_column_subsets(self, columns, min_length, max_length):
        """Helper function to randomly select subsets of columns"""
        # get random state object
        rng = self._rng

        # shuffle columns
        rng.shuffle(columns)

        # if length is not variable, use available function to split into equally sized arrays
        if min_length == max_length:
            n_subsets = int(np.ceil(self.n_columns_ / max_length))
            return np.array_split(columns, n_subsets)

        # otherwise iterate through columns, selecting uniformly random number of columns within
        # given bounds for each subset
        subsets = []
        it = iter(columns)  # iterator over columns
        while True:
            # draw random number of columns within bounds
            n_columns_in_subset = rng.random.randint(min_length, max_length + 1)

            # select number of columns and move iterator ahead
            subset = list(islice(it, n_columns_in_subset))

            # append if non-empty, otherwise break while loop
            if len(subset) > 0:
                subsets.append(np.array(subset))
            else:
                break

        return subsets

    def _random_instance_subset(self, y, n_instances, classes=None, bootstrap=False):
        """Select subset of instances (with replacements) conditional on random subset of classes"""
        # get random state object
        rng = self._rng

        # get random subset of classes if not given
        if classes is None:
            n_classes = rng.randint(1, len(self.classes_) + 1)
            classes = rng.choice(self.classes_, size=n_classes, replace=False)

        # get instances for selected classes
        isin_classes = np.where(np.isin(y, classes))[0]

        # if no bootstrap sample is taken (sampling with replacement), n_instances cannot be larger than number of
        # instances in selected classes
        if not bootstrap:
            n_instances = np.minimum(n_instances, len(isin_classes))

        # randomly select bootstrap subset of instances for selected classes
        instance_subset = rng.choice(isin_classes, size=n_instances, replace=bootstrap)
        return classes, instance_subset[:, None]

    def _preprocess_X(self, X):
        """Helper function to preprocess X including removal of constant columns and
         scaling"""
        # remove zero-variance/constant columns
        remover = VarianceThreshold(threshold=0)
        Xt = remover.fit_transform(X)

        # scale columns
        # other options for the scaler are StandardScaler(with_mean=True, with_std=True)
        # and MinMaxScaler()
        scaler = Normalizer()
        Xt = scaler.fit_transform(Xt)
        return Xt

    def predict_proba(self, X):
        """Predict probabilities"""
        check_is_fitted(self, '_is_fitted')

        # check input
        X = check_array(X)

        # preprocess data
        X_norm = self._preprocess_X(X)

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

    def predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : array-like or sparse matrix of shape = [n_samples, n_features]
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array of shape = [n_samples] or [n_samples, n_outputs]
            The predicted classes.
        """
        proba = self.predict_proba(X)

        if self.n_outputs_ == 1:
            return self.classes_.take(np.argmax(proba, axis=1), axis=0)

        else:
            n_samples = proba[0].shape[0]
            # all dtypes should be the same, so just take the first
            class_type = self.classes_[0].dtype
            predictions = np.empty((n_samples, self.n_outputs_),
                                   dtype=class_type)

            for k in range(self.n_outputs_):
                predictions[:, k] = self.classes_[k].take(np.argmax(proba[k],
                                                                    axis=1),
                                                          axis=0)
            return predictions
