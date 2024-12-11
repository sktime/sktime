"""Feature Selection Methods."""

__author__ = ["aykut-uz"]
__all__ = ["BaseFeatureSelection",
           "CoefficientFeatureSelection",
           "OMPFeatureSelection",
           "LassoFeatureSelection", 
           "XGBFeatureSelection"]

from sktime.transformations.base import BaseTransformer
import numpy as np
from sklearn.base import clone
import warnings

def get_top_n(number_list, n):
    return np.argsort(number_list)[-n:]

class BaseFeatureSelection(BaseTransformer):
    _tags = {
        "scitype:transform-input": "Series",
        # what is the scitype of X: Series, or Panel
        "scitype:transform-output": "Series",
        # what scitype is returned: Primitives, Series, Panel
        "scitype:instancewise": True,  # is this an instance-wise transform?
        "X_inner_mtype": ["pd.DataFrame", "pd.Series"],
        # which mtypes do _fit/_predict support for X?
        "y_inner_mtype": "pd.Series",  # which mtypes do _fit/_predict support for y?
        "fit_is_empty": False,
        "transform-returns-same-time-index": True,
        "skip-inverse-transform": True,
        "univariate-only": False,
    }
    def __init__(
            self,
            selector,
            n_features=6,
            random_state=None,
    ):
        self.n_features = n_features
        self.selector = selector
        self.random_state = random_state
        super().__init__()
    
    def _get_score(self,):
        """
        A private method that returns a score of a feature based on the feature selection technique.
        Returns
        -------
        A list of scores ordered according to features in the data matrix.
        """
        raise NotImplementedError

    def _fit(self, X, y=None):
        """
        Fit the transformer to X and y.
        Private _fit will be called in fit() and contains core logic
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
        self.selector_.fit(X, y)
        self.score_ = self._get_score()
        self.top_n_indices_ = get_top_n(self.score_, self.n_features_)
        self.selected_feature_names_ = [self.X_.columns[index] for index in self.top_n_indices_]
        self.selected_feature_scores_ = [self.score_[index] for index in self.top_n_indices_]
        return self
    
    def _transform(self, X, y=None):
        """
        Selects the feature columns in X according to the learned feature selection method.
        Parameters
        ----------
        X: pd.Series or pd.DataFrame
             Contains (exogenous) features.
        y: pd.Series or pd.DataFrame
             Contains the labels (target) of the time series.
        Returns
        -------
        Reduced data matrix X with the top n=n_features features selected by the feature selection method.
        """
        X_selected = X.loc[:, self.selected_feature_names_]
        return X_selected

class CoefficientFeatureSelection(BaseFeatureSelection):
    """A feature selection wrapper that is based on models with coefficients/weights (e.g., Lasso regression or Orthogonal Matching Pursuit).

    """
    def __init__(self, 
                n_features, 
                coefficient_selector, 
                coefficient_attribute="coef_"
    ):
        self.coefficient_attribute=coefficient_attribute
        super().__init__(
            selector=coefficient_selector,
            n_features=n_features
        )
    
    def _get_score(self):
        """
        Private method returns the score based on the absolute coefficients of the solution in the coefficient_selector regressions.
        Returns
        -------
        A list of absolute coefficients as scores ordered according to features in the data matrix.
        """
        try:
            coefficients = getattr(self.selector_, self.coefficient_attribute)
        except AttributeError:
            raise AttributeError(f"The passed selector of type {type(self.selector_)} has no attribute {self.coefficient_attribute}.")
        non_zero_scores = sum(coefficients != 0)
        if  non_zero_scores < self.n_features_:
            warnings.warn(
                f"""Lasso regression has {non_zero_scores} non-zero coefficients, but {self.n_features_} are required.
                {self.n_features_ - non_zero_scores} of the selected features have a zero coefficient. 
                """
            )
        return np.abs(coefficients)


class OMPFeatureSelection(CoefficientFeatureSelection):
    """
    Performs feature selection using Orthogonal Matching Pursuit (OMP).
    """
    def __init__(self, n_features):
        from sklearn.linear_model import OrthogonalMatchingPursuit
        super().__init__(
            selector=OrthogonalMatchingPursuit(n_nonzero_coefs=n_features), 
            n_features=n_features)


class LassoFeatureSelection(CoefficientFeatureSelection):
    """
    Performs feature selection using Lasso regressions.
    """
    def __init__(self, 
                n_features=10, 
                alpha=1.0,
                random_state=None,
    ):
        from sklearn.linear_model import Lasso
        self.alpha = alpha
        super().__init__(
            selector=Lasso(alpha=alpha, random_state=random_state),
            n_features=n_features
        )


class XGBFeatureSelection(BaseFeatureSelection):
    """
    Performs feature selection using XGBRegressors feature importance.
    """
    def __init__(
            self,
            n_features=10,
            importance_type="gain",
            model_params=None,
            random_state=None,
    ):

        from xgboost import XGBRegressor
        self.importance_type = importance_type
        self.model_params = dict() if model_params is None else model_params
        if random_state is None and not model_params is None:
            random_state = model_params.get("random_state", None)
        super().__init__(selector=XGBRegressor(random_state=random_state, **self.model_params),
                         n_features=n_features)

    def _get_score(self):
        """
        Private method returns the score based on the scoring of the XGB regression.
        The scoring depends on the importance_type set during initialization (see importance_type in
        XGBRegressor for more details).
        Returns
        -------
        A list of scores ordered according to features in the data matrix.
        Note that, in contrast to XGB, also zero score features are inlcuded in the list.
        """
        booster = self.selector_.get_booster()
        scores = booster.get_score(importance_type=self.importance_type)
        # Note: get_score() does not include zero score features!
        importance_scores_all = np.zeros(self.X_.shape[0])
        for i, feature in enumerate(self.X_.columns):
            if feature in scores:
                importance_scores_all[i] = scores[feature]
        return importance_scores_all