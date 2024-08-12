"""Bagging time series classifiers."""

__author__ = ["fkiraly"]
__all__ = ["BaggingClassifier"]

from math import ceil

import numpy as np
import pandas as pd

from sktime.classification.base import BaseClassifier


class BaggingClassifier(BaseClassifier):
    """Bagging ensemble of time series classifiers.

    Fits ``n_estimators`` clones of a classifier on
    datasets which are instance sub-samples and/or variable sub-samples.

    On ``predict_proba``, the mean average of probabilistic predictions is returned.
    For a deterministic classifier, this results in majority vote for ``predict``.

    The estimator allows to choose sample sizes for instances, variables,
    and whether sampling is with or without replacement.

    Direct generalization of ``sklearn``'s ``BaggingClassifier``
    to the time series classification task.

    Note: if ``n_features=1``, ``BaggingClassifier`` turns a univariate classifier into
    a multivariate classifier, because slices seen by ``estimator`` are all univariate.
    This can be used to give a univariate classifier multivariate capabilities.

    Parameters
    ----------
    estimator : sktime classifier, descendant of BaseClassifier
        classifier to use in the bagging estimator
    n_estimators : int, default=10
        number of estimators in the sample for bagging
    n_samples : int or float, default=1.0
        The number of instances drawn from ``X`` in ``fit`` to train each clone
        If int, then indicates number of instances precisely
        If float, interpreted as a fraction, and rounded by ``ceil``
    n_features : int or float, default=1.0
        The number of features/variables drawn from ``X`` in ``fit`` to train each clone
        If int, then indicates number of instances precisely
        If float, interpreted as a fraction, and rounded by ``ceil``
        Note: if n_features=1, BaggingClassifier turns a univariate classifier into
        a multivariate classifier (as slices seen by ``estimator`` are all univariate).
    bootstrap : boolean, default=True
        whether samples/instances are drawn with replacement (True) or not (False)
    bootstrap_features : boolean, default=False
        whether features/variables are drawn with replacement (True) or not (False)
    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number generator;
        If ``RandomState`` instance, ``random_state`` is the random number generator;
        If None, the random number generator is the ``RandomState`` instance used
        by ``np.random``.

    Attributes
    ----------
    estimators_ : list of of sktime classifiers
        clones of classifier in ``estimator`` fitted in the ensemble

    Examples
    --------
    >>> from sktime.classification.ensemble import BaggingClassifier
    >>> from sktime.classification.kernel_based import RocketClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train") # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test") # doctest: +SKIP
    >>> clf = BaggingClassifier(
    ...     RocketClassifier(num_kernels=100),
    ...     n_estimators=10,
    ... ) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    BaggingClassifier(...)
    >>> y_pred = clf.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["fkiraly"],
        # estimator type
        # --------------
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "X_inner_mtype": ["pd-multiindex", "nested_univ"],
    }

    def __init__(
        self,
        estimator,
        n_estimators=10,
        n_samples=1.0,
        n_features=1.0,
        bootstrap=True,
        bootstrap_features=False,
        random_state=None,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.n_samples = n_samples
        self.n_features = n_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.random_state = random_state

        super().__init__()

        if n_features == 1:
            # if n_features == 1, this turns a univariate classifier into multivariate
            tags_to_clone = ["capability:missing_values"]
        else:
            tags_to_clone = ["capability:multivariate", "capability:missing_values"]
        self.clone_tags(estimator, tags_to_clone)

    def _fit(self, X, y):
        """Fit time series classifier to training data.

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : 1D np.array of int, of shape [n_instances] - class labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.
        """
        estimator = self.estimator
        n_estimators = self.n_estimators
        n_samples = self.n_samples
        n_features = self.n_features
        bootstrap = self.bootstrap
        bootstrap_ft = self.bootstrap_features
        random_state = self.random_state
        np.random.seed(random_state)

        if isinstance(X.index, pd.MultiIndex):
            inst_ix = X.index.droplevel(-1).unique()
        else:
            inst_ix = X.index
        col_ix = X.columns
        n = len(inst_ix)
        m = len(col_ix)

        if isinstance(n_samples, float):
            n_samples_ = ceil(n_samples * n)
        else:
            n_samples_ = n_samples

        if isinstance(n_features, float):
            n_features_ = ceil(n_features * m)
        else:
            n_features_ = n_features

        self.estimators_ = []
        self._col_ixis = []
        for _i in range(n_estimators):
            esti = estimator.clone()
            row_iloc = pd.RangeIndex(n)
            row_ss = _random_ss_ix(row_iloc, size=n_samples_, replace=bootstrap)
            inst_ix_i = inst_ix[row_ss]
            col_ix_i = _random_ss_ix(col_ix, size=n_features_, replace=bootstrap_ft)
            # if we bootstrap, we need to take care to ensure the
            # indices end up unique
            if not isinstance(X.index, pd.MultiIndex):
                Xi = X.loc[inst_ix_i, col_ix_i]
                Xi = Xi.reset_index(drop=True)
            else:
                Xis = [X.loc[[ix], col_ix_i].droplevel(0) for ix in inst_ix_i]
                Xi = pd.concat(Xis, keys=pd.RangeIndex(len(inst_ix_i)))

            if bootstrap_ft:
                Xi.columns = pd.RangeIndex(len(col_ix_i))

            yi = y[row_ss]
            self.estimators_ += [esti.fit(Xi, yi)]
            self._col_ixis += [col_ix_i]

        return self

    def _predict_proba(self, X) -> np.ndarray:
        """Predicts labels probabilities for sequences in X.

        Parameters
        ----------
        X : 3D np.array of shape = [n_instances, n_dimensions, series_length]
            The data to make predict probabilities for.

        Returns
        -------
        y : array-like, shape = [n_instances, n_classes_]
            Predicted probabilities using the ordering in classes_.
        """
        classes = pd.Index(self.classes_)

        y_probas = []
        for esti, col_ix_i in zip(self.estimators_, self._col_ixis):
            Xi = X.loc[:, col_ix_i]
            if self.bootstrap_features:
                Xi.columns = pd.RangeIndex(len(col_ix_i))

            y_probas += [esti.predict_proba(Xi)]

        est_shape = (len(y_probas[0]), len(classes))
        y_proba_np = np.zeros((len(y_probas), est_shape[0], est_shape[1]))
        y_proba_np = np.zeros((est_shape[0], est_shape[1], len(y_probas)))

        for i, y_proba in enumerate(y_probas):
            cls_ix = self.estimators_[i].classes_
            ixer = classes.get_indexer_for(cls_ix)
            y_proba_np[:, ixer, i] = y_proba

        y_proba = np.mean(y_proba_np, axis=2)

        return y_proba

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``.
        """
        from sktime.classification.feature_based import SummaryClassifier

        params1 = {"estimator": SummaryClassifier()}
        params2 = {
            "estimator": SummaryClassifier(),
            "n_samples": 0.5,
            "n_features": 0.5,
        }
        params3 = {
            "estimator": SummaryClassifier(),
            "n_samples": 7,
            "n_features": 2,
            "bootstrap": False,
            "bootstrap_features": True,
        }

        # force-create a classifier that cannot handle multivariate
        univariate_dummy = SummaryClassifier()
        univariate_dummy.set_tags(**{"capability:multivariate": False})
        # this should still result in a multivariate classifier
        params4 = {
            "estimator": univariate_dummy,
            "n_features": 1,
        }

        return [params1, params2, params3, params4]


def _random_ss_ix(ix, size, replace=True):
    a = range(len(ix))
    ixs = ix[np.random.choice(a, size=size, replace=replace)]
    return ixs
