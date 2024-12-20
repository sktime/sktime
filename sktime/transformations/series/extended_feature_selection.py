"""Feature Selection Methods."""

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.transformations.base import BaseTransformer

__author__ = ["aykut-uz"]
__all__ = [
    "BaseFeatureSelection",
    "CoefficientFeatureSelection",
    "OMPFeatureSelection",
    "LassoFeatureSelection",
    "XGBFeatureSelection",
]


def get_top_n(number_list, n):
    return np.argsort(number_list)[-n:]


class BaseFeatureSelection(BaseTransformer, metaclass=ABCMeta):
    """Abstract class for feature selection defining fit and transform methods logic.

    The class uses a specified selector to perform feature selection.
    It fits the selector to input data, calculates scores for each feature
    using the _get_score method (to be implemented by child classes),
    and stores the top n features and their respective scores.

    Attributes
    ----------
    n_features : int
        Number of features to select.
    selector : object
        Feature selection algorithm to use.
    random_state : int, optional
        Random state for reproducibility.

    Methods
    -------
    _get_score()
        Abstract method to calculate scores for each feature in a dataset.
    _fit(X, y=None)
        Fits the transformer to X and y.
    _transform(X, y=None)
        Transforms the input data by selecting top n features.

    """

    _tags = {
        "authors": ["aykut-uz"],
        "maintainers": ["aykut-uz"],
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:instancewise": True,
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        "y_inner_mtype": "pd.Series",
        "fit_is_empty": False,
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,
        "univariate-only": False,
    }

    def __init__(
        self,
        selector,
        n_features,
        random_state=None,
    ):
        if not isinstance(n_features, (int, np.integer)) or n_features < 1:
            raise ValueError(
                f"n_features must be positive integer but was {n_features}"
            )

        self.n_features = n_features
        self.selector = selector
        self.random_state = random_state
        super().__init__()

    @abstractmethod
    def _get_score(
        self,
    ):
        """Core logic to obtain the scores of features.

        A private method that returns a score of a feature based on the
        feature selection technique.

        Returns
        -------
        A list of scores ordered according to features in the data matrix.

        """
        pass

    def _fit(self, X, y=None):
        """Core logic of fit.

        Fit the transformer to X and y.
        This method saves the input data, clones the selector, fits the selector to the
        data, calculates feature scores and stores the top n feature names and their
        respective scores.

        Parameters
        ----------
        X: pd.Series or pd.DataFrame
            Contains (exogenous) features.
        y: pd.Series or pd.DataFrame
            Contains the labels (target) of the time series.

        Returns
        -------
        self: a fitted transformer.
        """
        self.X_ = X
        self.y_ = y
        self.n_features_ = self.n_features
        self.selector_ = clone(self.selector)
        self.score_ = None
        self.top_n_indices_ = None
        self.selected_feature_names_ = None
        self.selected_feature_scores_ = None
        if isinstance(X, pd.Series):
            return self
        if X.shape[1] < self.n_features_:
            self.n_features_ = X.shape[1]
            warnings.warn(
                f"Requested number of selected features is {self.n_features}, "
                f"but data matrix X has only {X.shape[1]} features. "
                f"n_features is set to {X.shape[1]}."
            )
        self.selector_.fit(X, y)
        self.score_ = self._get_score()
        self.top_n_indices_ = get_top_n(self.score_, self.n_features_)
        self.selected_feature_names_ = [
            self.X_.columns[index] for index in self.top_n_indices_
        ]
        self.selected_feature_scores_ = [
            self.score_[index] for index in self.top_n_indices_
        ]
        return self

    def _transform(self, X, y=None):
        """Core logic of transform.

        Selects the feature columns in X according to the learned
        feature selection method.

        Parameters
        ----------
        X: pd.Series or pd.DataFrame
             Contains (exogenous) features.
        y: pd.Series or pd.DataFrame
             Contains the labels (target) of the time series.

        Returns
        -------
        Reduced data matrix X with the top n=n_features features selected by
        the feature selection method.
        """
        if isinstance(X, pd.Series):
            return X
        X_selected = X.loc[:, self.selected_feature_names_]
        return X_selected


class CoefficientFeatureSelection(BaseFeatureSelection):
    """A feature selection wrapper for coefficient-based feature selection.

    This class selects features based on the absolute value of their coefficient in a
    regression model. The specific regression model is provided via
    the `selector` parameter.

    Parameters
    ----------
    n_features : int
        Number of features to select.
    selector : object
        The regression model to use as a base for feature selection.
    coefficient_attribute : str, optional
        Name of the attribute of the 'selector' object that holds the coefficients.
        Defaults to "coef_".
    random_state : int, RandomState instance or None, optional
        Controls the randomness of the estimator. Default is None.
    """

    def __init__(
        self, n_features, selector, coefficient_attribute="coef_", random_state=None
    ):
        self.coefficient_attribute = coefficient_attribute
        super().__init__(
            selector=selector, n_features=n_features, random_state=random_state
        )

    def _get_score(self):
        """Return scores based on the coefficient attribute of the regressor.

        Private method returns the score based on the absolute coefficients of
        the solution in the coefficient_selector regressions.

        Returns
        -------
        A list of absolute coefficients as scores ordered according to features
        in the data matrix.
        """
        try:
            coefficients = np.asarray(
                getattr(self.selector_, self.coefficient_attribute)
            )
        except AttributeError:
            raise AttributeError(
                f"The passed selector of type {type(self.selector_)} has "
                f" no attribute {self.coefficient_attribute}."
            )
        number_non_zero_scores = np.count_nonzero(coefficients)
        if number_non_zero_scores < self.n_features_:
            warnings.warn(
                f"Selector {type(self.selector_).__name__} has "
                f"{number_non_zero_scores} non-zero coefficients "
                f"but {self.n_features_} are required. "
                f"{self.n_features_ - number_non_zero_scores} of the "
                f"selected features have a zero coefficient."
            )
        return np.abs(coefficients)

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
        from sklearn.linear_model import Lasso, OrthogonalMatchingPursuit

        params = [
            {"n_features": 1, "selector": OrthogonalMatchingPursuit()},
            {"n_features": 3, "selector": Lasso()},
        ]
        return params


class OMPFeatureSelection(CoefficientFeatureSelection):
    """Perform feature selection using Orthogonal Matching Pursuit (OMP).

    The Orthogonal Matching Pursuit algorithm approximates the fit of a
    linear model with constraints imposed on the number of non-zero coefficients
    (i.e., the L0 pseudo-norm).

    Parameters
    ----------
    n_features : int
        Number of non-zero coefficients to target in the approximation.

    Examples
    --------
    >>> from sktime.transformations.series.extended_feature_selection import (
    ...     OMPFeatureSelection
    ...     )
    >>> from sktime.datasets import load_longley
    >>> y, X = load_longley()
    >>> feature_selector = OMPFeatureSelection(n_features=3)
    >>> X_transformed = feature_selector.fit_transform(X, y)
    """

    _tags = {
        **BaseFeatureSelection._tags,
        "python_dependencies": ["scikit-learn"],
    }

    def __init__(self, n_features):
        from sklearn.linear_model import OrthogonalMatchingPursuit

        super().__init__(
            selector=OrthogonalMatchingPursuit(n_nonzero_coefs=n_features),
            n_features=n_features,
        )

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
        params = [{"n_features": 1}, {"n_features": 10}]
        return params


class LassoFeatureSelection(CoefficientFeatureSelection):
    """Perform feature selection using Lasso regressions.

    Parameters
    ----------
    n_features : int
        The number of features to select.
    alpha : float, default=1.0
        The regularization strength of the Lasso regression.
    random_state : int or RandomState instance, default=None
        Controls the randomness of the estimator.

    Examples
    --------
    >>> from sktime.transformations.series.extended_feature_selection import (
    ...     LassoFeatureSelection
    ...     )
    >>> from sktime.datasets import load_longley
    >>> y, X = load_longley()
    >>> feature_selector = LassoFeatureSelection(n_features=3,
    ...                     alpha=2.0, random_state=10)
    >>> X_transformed = feature_selector.fit_transform(X, y)
    """

    _tags = {
        **BaseFeatureSelection._tags,
        "python_dependencies": ["scikit-learn"],
    }

    def __init__(self, n_features, **model_params):
        from sklearn.linear_model import Lasso

        self.model_params = model_params
        super().__init__(
            selector=Lasso(**self.model_params),
            n_features=n_features,
            random_state=model_params.get("random_state", None),
        )

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
        params = [
            {"n_features": 1, "alpha": 2.0},
            {
                "n_features": 1,
                "random_state": 10,
                "alpha": 3.0,
            },
        ]
        return params


class XGBFeatureSelection(BaseFeatureSelection):
    """Perform feature selection using XGBRegressors feature importance.

    Parameters
    ----------
    n_features : int
        The number of features to select.
    importance_type : str, optional
        The feature importance type, by default "gain".
    model_params : dict, optional
        The parameters for the model, by default None.
    random_state : int, optional
        The seed used by the random number generator, by default None.

    Attributes
    ----------
    importance_type : str
        The feature importance type.
    model_params : dict
        The parameters for the model.
    model_params_ : dict
        A copy of the model parameters.

    Examples
    --------
    >>> from sktime.transformations.series.extended_feature_selection import (
    ...     XGBFeatureSelection
    ...     )
    >>> from sktime.datasets import load_longley
    >>> y, X = load_longley()
    >>> model_params = {"n_estimators":100}
    >>> feature_selector = XGBFeatureSelection(n_features=3, model_params=model_params)
    >>> X_transformed = feature_selector.fit_transform(X, y)
    """

    _tags = {
        **BaseFeatureSelection._tags,
        "python_dependencies": ["xgboost"],
    }

    def __init__(self, n_features, importance_type="gain", **model_params):
        from xgboost import XGBRegressor

        self.importance_type = importance_type
        self.model_params = model_params
        super().__init__(
            selector=XGBRegressor(**self.model_params),
            n_features=n_features,
            random_state=model_params.get("random_state", None),
        )

    def _get_score(self):
        """Return scores according to XGB regressor.

        Private method returns the score based on the scoring of the XGB regression.
        The scoring depends on the importance_type set during initialization
        (see importance_type in XGBRegressor for more details).

        Returns
        -------
        A list of scores ordered according to features in the data matrix.
        Note that, in contrast to XGB, also zero score features are inlcuded in the
        list.
        """
        booster = self.selector_.get_booster()
        scores = booster.get_score(importance_type=self.importance_type)
        # Note: get_score() does not include zero score features!
        importance_scores_all = np.zeros(self.X_.shape[1])
        for i, feature in enumerate(self.X_.columns):
            if feature in scores:
                importance_scores_all[i] = scores[feature]
        return importance_scores_all

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
        params = [
            {"n_features": 1},
            {"n_features": 1, "n_estimators": 14, "random_state": 10},
        ]
        return params
