# -*- coding: utf-8 -*-
"""Bagging time series classifiers."""

__author__ = ["fkiraly"]
__all__ = ["BaggingClassifier"]

from math import ceil

import numpy as np
import pandas as pd

from sktime.classification.base import BaseClassifier


class BaggingClassifier(BaseClassifier):
    """Weighted ensemble of classifiers with fittable ensemble weight.

    Produces a probabilistic prediction which is the weighted average of
    predictions of individual classifiers.
    Classifier with name `name` has ensemble weight in `weights_[name]`.
    `weights_` is fitted in `fit`, if `weights` is a scalar, otherwise fixed.

    If `weights` is a scalar, empirical training loss is computed for each classifier.
    In this case, ensemble weights of classifier is empirical loss,
    to the power of `weights` (a scalar).

    The evaluation for the empirical training loss can be selected
    through the `metric` and `metric_type` parameters.

    The in-sample empirical training loss is computed in-sample or out-of-sample,
    depending on the `cv` parameter. None = in-sample; other = cross-validated oos.

    Parameters
    ----------
    classifiers : dict or None, default=None
        Parameters for the ShapeletTransformClassifier module. If None, uses the
        default parameters with a 2 hour transform contract.
    weights : float, or iterable of float, optional, default=None
        if float, ensemble weight for classifier i will be train score to this power
        if iterable of float, must be equal length as classifiers
            ensemble weight for classifier i will be weights[i]
        if None, ensemble weights are equal (uniform average)
    cv : None, int, or sklearn cross-validation object, optional, default=None
        determines whether in-sample or which cross-validated predictions used in fit
        None : predictions are in-sample, equivalent to fit(X, y).predict(X)
        cv : predictions are equivalent to fit(X_train, y_train).predict(X_test)
            where multiple X_train, y_train, X_test are obtained from cv folds
            returned y is union over all test fold predictions
            cv test folds must be non-intersecting
        int : equivalent to cv=KFold(cv, shuffle=True, random_state=x),
            i.e., k-fold cross-validation predictions out-of-sample
            random_state x is taken from self if exists, otherwise x=None
    metric : sklearn metric for computing training score, default=accuracy_score
        only used if weights is a float
    metric_type : str, one of "point" or "proba", default="point"
        type of sklearn metric, point prediction ("point") or probabilistic ("proba")
        if "point", most probable class is passed as y_pred
        if "proba", probability of most probable class is passed as y_pred
    random_state : int or None, default=None
        Seed for random number generation.

    Attributes
    ----------
    classifiers_ : list of tuples (str, classifier) of sktime classifiers
        clones of classifies in `classifiers` which are fitted in the ensemble
        is always in (str, classifier) format, even if `classifiers` is just a list
        strings not passed in `classifiers` are replaced by unique generated strings
        i-th classifier in `classifier_` is clone of i-th in `classifier`
    weights_ : dict with str being classifier names as in `classifiers_`
        value at key is ensemble weights of classifier with name key
        ensemble weights are fitted in `fit` if `weights` is a scalar

    Examples
    --------
    >>> from sktime.classification.dummy import DummyClassifier
    >>> from sktime.classification.kernel_based import RocketClassifier
    >>> from sktime.datasets import load_unit_test
    >>> X_train, y_train = load_unit_test(split="train") # doctest: +SKIP
    >>> X_test, y_test = load_unit_test(split="test") # doctest: +SKIP
    >>> clf = WeightedEnsembleClassifier(
    ...     [DummyClassifier(), RocketClassifier(num_kernels=100)],
    ...     weights=2,
    ... ) # doctest: +SKIP
    >>> clf.fit(X_train, y_train) # doctest: +SKIP
    WeightedEnsembleClassifier(...)
    >>> y_pred = clf.predict(X_test) # doctest: +SKIP
    """

    _tags = {
        "capability:multivariate": True,
        "capability:missing_values": True,
        "capability:predict_proba": True,
        "X_inner_mtype": ["pd-multiindex", "nested_univ"]
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

        super(BaggingClassifier, self).__init__()

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
        for i in range(n_estimators):
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
        y_probas = [est.predict_proba(X) for est in self.estimators_]
        y_proba = np.mean(y_probas, axis=0)

        return y_proba

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            For classifiers, a "default" set of parameters should be provided for
            general testing, and a "results_comparison" set for comparing against
            previously recorded results if the general set does not produce suitable
            probabilities to compare against.

        Returns
        -------
        params : dict or list of dict, default={}
            Parameters to create testing instances of the class.
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`.
        """
        from sktime.classification.dummy import DummyClassifier

        params1 = {"estimator": DummyClassifier()}
        params2 = {
            "estimator": DummyClassifier(),
            "n_samples": 0.5,
            "n_features": 0.5,
        }
        params3 = {
            "estimator": DummyClassifier(),
            "n_samples": 7,
            "n_features": 2,
            "bootstrap": False,
            "bootstrap_features": True,
        }

        return [params1, params2, params3]


def _random_ss_ix(ix, size, replace=True):
    a = range(len(ix))
    ixs = ix[np.random.choice(a, size=size, replace=replace)]
    return ixs
