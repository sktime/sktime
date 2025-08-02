#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements a composite Time series Forest Regressor that accepts a pipeline."""

__author__ = ["mloning", "AyushmaanSeth"]
__all__ = ["ComposableTimeSeriesForestRegressor"]

import numbers

import numpy as np
from sklearn.ensemble._base import _partition_estimators
from sklearn.ensemble._forest import (
    _generate_unsampled_indices,
    _get_n_samples_bootstrap,
)
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from sktime.base._panel.forest._composable import BaseTimeSeriesForest
from sktime.regression.base import BaseRegressor
from sktime.transformations.panel.summarize import RandomIntervalFeatureExtractor
from sktime.utils.slope_and_trend import _slope
from sktime.utils.validation.panel import check_X, check_X_y
from sktime.utils.warnings import warn


class ComposableTimeSeriesForestRegressor(BaseTimeSeriesForest, BaseRegressor):
    """Time-Series Forest Regressor.

    A time series forest is a meta estimator and an adaptation of the random
    forest for time-series/panel data that fits a number of decision tree
    regressors on various sub-samples of a transformed dataset and uses
    averaging to improve the predictive accuracy and control over-fitting.
    The sub-sample size is always the same as the original input sample size
    but the samples are drawn with replacement if ``bootstrap=True`` (default).

    Parameters
    ----------
    estimator : Pipeline
        A pipeline consisting of series-to-tabular transformations
        and a decision tree regressor as final estimator.
    n_estimators : integer, optional (default=100)
        The number of trees in the forest.
    criterion : string, optional (default="squared_error")
        The function to measure the quality of a split. Supported criteria are
        "squared_error" for the mean squared error, which is equal to variance reduction
        as feature selection criterion and minimizes the L2 loss using the mean of each
        terminal node, "friedman_mse", which uses mean squared error with Friedman's
        improvement score for potential splits, "absolute_error" for the mean absolute
        error, which minimizes the L1 loss using the median of each terminal node,
        and "poisson" which uses reduction in Poisson deviance to find splits.
    max_depth : integer or None, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, float, optional (default=2)
        The minimum number of samples required to split an internal node:
        - If int, then consider ``min_samples_split`` as the minimum number.
        - If float, then ``min_samples_split`` is a fraction and
          ``ceil(min_samples_split * n_samples)`` are the minimum
          number of samples for each split.
    min_samples_leaf : int, float, optional (default=1)
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider ``min_samples_leaf`` as the minimum number.
        - If float, then ``min_samples_leaf`` is a fraction and
          ``ceil(min_samples_leaf * n_samples)`` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, optional (default=0.)
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float, string or None, optional (default="auto")
        The number of features to consider when looking for the best split:
        - If int, then consider ``max_features`` features at each split.
        - If float, then ``max_features`` is a fraction and
          ``int(max_features * n_features)`` features are considered at each
          split.
        - If "auto", then ``max_features=sqrt(n_features)``.
        - If "sqrt", then ``max_features=sqrt(n_features)`` (same as "auto").
        - If "log2", then ``max_features=log2(n_features)``.
        - If None, then ``max_features=n_features``.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    max_leaf_nodes : int or None, optional (default=None)
        Grow trees with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    min_impurity_decrease : float, optional (default=0.)
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
    bootstrap : boolean, optional (default=True)
        Whether bootstrap samples are used when building trees.
    oob_score : bool (default=False)
        Whether to use out-of-bag samples to estimate
        the generalization accuracy.
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for both ``fit`` and ``predict``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors.
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``.
    verbose : int, optional (default=0)
        Controls the verbosity when fitting and predicting.
    warm_start : bool, optional (default=False)
        When set to ``True``, reuse the solution of the previous call to fit
        and add more estimators to the ensemble, otherwise, just fit a whole
        new forest.
    None, optional (default=None)
        Weights associated with classes in the form ``{class_label: weight}``.
        If not given, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        The "balanced_subsample" mode is the same as "balanced" except that
        weights are computed based on the bootstrap sample for every tree
        grown.
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The collection of fitted sub-estimators.
    n_features_ : int
        The number of features when ``fit`` is performed.
    n_outputs_ : int
        The number of outputs when ``fit`` is performed.
    feature_importances_ : array of shape = [n_features]
        The feature importances (the higher, the more important the feature).
    oob_score_ : float
        Score of the training dataset obtained using an out-of-bag estimate.
    oob_decision_function_ : array of shape = [n_samples, n_classes]
        Decision function computed with out-of-bag estimate on the training
        set. If n_estimators is small it might be possible that a data point
        was never left out during the bootstrap. In this case,
        ``oob_decision_function_`` might contain NaN.
    class_weight: dict, list of dicts, "balanced", "balanced_subsample" or \
        None, optional (default=None)
        Not needed here, added in the constructor to align with base class \
            sharing both Classifier and Regressor parameters.
    """

    _tags = {
        "python_dependencies": ["joblib"],
        "X_inner_mtype": "nested_univ",  # nested pd.DataFrame
    }

    def __init__(
        self,
        estimator=None,
        n_estimators=100,
        criterion="squared_error",
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0.0,
        max_features=None,
        max_leaf_nodes=None,
        min_impurity_decrease=0.0,
        bootstrap=False,
        oob_score=False,
        n_jobs=None,
        random_state=None,
        verbose=0,
        warm_start=False,
        max_samples=None,
    ):
        self.estimator = estimator
        # Assign values, even though passed on to base estimator below,
        # necessary here for cloning
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.max_samples = max_samples

        # Pass on params.
        super().__init__(
            base_estimator=None,
            n_estimators=n_estimators,
            estimator_params=None,
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            max_samples=max_samples,
        )
        BaseRegressor.__init__(self)

        # We need to add is-fitted state when inheriting from scikit-learn
        self._is_fitted = False

    def fit(self, X, y, **kwargs):
        """Wrap fit to call BaseRegressor.fit.

        This is a fix to get around the problem with multiple inheritance. The problem
        is that if we just override _fit, this class inherits the fit from the sklearn
        class BaseTimeSeriesForest. This is the simplest solution, albeit a little
        hacky.
        """
        return BaseRegressor.fit(self, X=X, y=y, **kwargs)

    def predict(self, X, **kwargs) -> np.ndarray:
        """Wrap predict to call BaseRegressor.predict."""
        return BaseRegressor.predict(self, X=X, **kwargs)

    def predict_proba(self, X, **kwargs) -> np.ndarray:
        """Wrap predict_proba to call BaseRegressor.predict_proba."""
        return BaseRegressor.predict_proba(self, X=X, **kwargs)

    def _repr_html_inner(self):
        """Return HTML representation of class.

        This function is returned by the @property `_repr_html_` to make
        `hasattr(BaseObject, "_repr_html_") return `True` or `False` depending
        on `self.get_config()["display"]`.
        """
        return BaseRegressor._repr_html_inner(self)

    def _repr_mimebundle_(self, **kwargs):
        """Mime bundle used by jupyter kernels to display instances of BaseObject."""
        return BaseRegressor._repr_mimebundle_(self, **kwargs)

    def _fit(self, X, y):
        BaseTimeSeriesForest._fit(self, X=X, y=y)

    def _validate_estimator(self):
        if not isinstance(self.n_estimators, numbers.Integral):
            raise ValueError(
                f"n_estimators must be an integer, got {type(self.n_estimators)}."
            )

        if self.n_estimators <= 0:
            raise ValueError(
                f"n_estimators must be greater than zero, got {self.n_estimators}."
            )

        # Set base estimator
        if self.estimator is None:
            # Set default time series forest
            features = [np.mean, np.std, _slope]
            steps = [
                (
                    "transform",
                    RandomIntervalFeatureExtractor(
                        n_intervals="sqrt",
                        features=features,
                        random_state=self.random_state,
                    ),
                ),
                ("clf", DecisionTreeRegressor(random_state=self.random_state)),
            ]
            self._estimator = Pipeline(steps)

        else:
            # else check given estimator is a pipeline with prior
            # transformations and final decision tree
            if not isinstance(self.estimator, Pipeline):
                raise ValueError("`estimator` must be pipeline with transforms.")
            if not isinstance(self.estimator.steps[-1][1], DecisionTreeRegressor):
                raise ValueError(
                    "Last step in `estimator` must be DecisionTreeRegressor."
                )
            self._estimator = self.estimator

        # Set parameters according to naming in pipeline
        estimator_params = {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_weight_fraction_leaf": self.min_weight_fraction_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
        }
        final_estimator = self._estimator.steps[-1][0]
        self.estimator_params = {
            f"{final_estimator}__{pname}": pval
            for pname, pval in estimator_params.items()
        }

        # Set renamed estimator parameters
        for pname, pval in self.estimator_params.items():
            self.__setattr__(pname, pval)

    def _predict(self, X):
        """Predict class for X.

        The predicted class of an input sample is a vote by the trees in
        the forest, weighted by their probability estimates. That is,
        the predicted class is the one with highest mean probability
        estimate across the trees.

        Parameters
        ----------
        X : array-like or sparse matrix of shape (n_samples, n_features)
            The input samples. Internally, its dtype will be converted to
            ``dtype=np.float32``. If a sparse matrix is provided, it will be
            converted into a sparse ``csr_matrix``.

        Returns
        -------
        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        from joblib import Parallel, delayed

        X = check_X(X, enforce_univariate=True)

        X = self._validate_X_predict(X)

        # Assign chunk of trees to jobs
        n_jobs, _, _ = _partition_estimators(self.n_estimators, self.n_jobs)

        # Parallel loop
        y_hat = Parallel(n_jobs=n_jobs, verbose=self.verbose)(
            delayed(e.predict)(X, check_input=True) for e in self.estimators_
        )

        return np.sum(y_hat, axis=0) / len(self.estimators_)

    def _set_oob_score(self, X, y):
        """Compute out-of-bag scores."""
        X, y = check_X_y(X, y, enforce_univariate=True)

        n_samples = y.shape[0]

        predictions = np.zeros((n_samples, self.n_outputs_))
        n_predictions = np.zeros((n_samples, self.n_outputs_))

        n_samples_bootstrap = _get_n_samples_bootstrap(n_samples, self.max_samples)

        for estimator in self.estimators_:
            final_estimator = estimator.steps[-1][1]
            unsampled_indices = _generate_unsampled_indices(
                final_estimator.random_state, n_samples, n_samples_bootstrap
            )
            p_estimator = estimator.predict(X[unsampled_indices, :], check_input=False)

            if self.n_outputs_ == 1:
                p_estimator = p_estimator[:, np.newaxis]

            predictions[unsampled_indices, :] += p_estimator
            n_predictions[unsampled_indices, :] += 1

        if (n_predictions == 0).any():
            warn(
                "Some inputs in ComposableTimeSeriesRegressor do not have OOB scores. "
                "This probably means too few trees were used "
                "to compute any reliable oob estimates.",
                obj=self,
            )
            n_predictions[n_predictions == 0] = 1

        predictions /= n_predictions
        self.oob_prediction_ = predictions

        if self.n_outputs_ == 1:
            self.oob_prediction_ = self.oob_prediction_.reshape((n_samples,))

        self.oob_score_ = 0.0

        for k in range(self.n_outputs_):
            self.oob_score_ += r2_score(y[:, k], predictions[:, k])

        self.oob_score_ /= self.n_outputs_

    # TODO - Implement this abstract method properly.
    def _set_oob_score_and_attributes(self, X, y):
        raise NotImplementedError("Not implemented.")

    def _validate_y_class_weight(self, y):
        # in regression, we don't validate class weights
        # TODO remove from regression
        return y, None

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            ``MyClass(**params)`` or ``MyClass(**params[i])`` creates a valid test
            instance.
            ``create_test_instance`` uses the first (or only) dictionary in ``params``
        """
        param1 = {"n_estimators": 3}
        param2 = {
            "n_estimators": 5,
            "max_depth": 5,
            "min_samples_split": 4,
            "random_state": 42,
        }
        param3 = {
            "n_estimators": 10,
            "max_depth": 7,
            "min_samples_split": 0.2,
        }
        return [param1, param2, param3]
