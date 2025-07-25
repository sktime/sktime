"""Delegator mixin that delegates all methods to wrapped regressors.

Useful for building estimators where all but one or a few methods are delegated. For
that purpose, inherit from this estimator and then override only the methods     that
are not delegated.
"""

# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["fkiraly"]
__all__ = ["_DelegatedRegressor"]

from sktime.regression.base import BaseRegressor


class _DelegatedRegressor(BaseRegressor):
    """Delegator mixin that delegates all methods to wrapped regressor.

    Delegates inner regressor methods to a wrapped estimator.
        Wrapped estimator is value of attribute with name self._delegate_name.
        By default, this is "estimator_", i.e., delegates to self.estimator_
        To override delegation, override _delegate_name attribute in child class.

    Delegates the following inner underscore methods:
        _fit, _predict, _predict_proba

    Does NOT delegate get_params, set_params.
        get_params, set_params will hence use one additional nesting level by default.

    Does NOT delegate or copy tags, this should be done in a child class if required.
    """

    # attribute for _DelegatedRegressor, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedRegressor docstring
    _delegate_name = "estimator_"

    def _get_delegate(self):
        return getattr(self, self._delegate_name)

    def _set_delegated_tags(self, delegate=None):
        """Set delegated tags, only tags for boilerplate control.

        Writes tags to self.
        Can be used by descendant classes to set dependent tags.
        Makes safe baseline assumptions about tags, which can be overwritten.

        * data mtype tags are set to the most general value.
          This is to ensure that conversion is left to the inner estimator.
        * packaging tags such as "author" or "python_dependencies" are not cloned.
        * other boilerplate tags are cloned.

        Parameters
        ----------
        delegate : object, optional (default=None)
            object to get tags from, if None, uses self._get_delegate()

        Returns
        -------
        self : reference to self
        """
        from sktime.datatypes import MTYPE_LIST_PANEL, MTYPE_LIST_TABLE

        if delegate is None:
            delegate = self._get_delegate()

        TAGS_TO_DELEGATE = [
            "capability:multioutput",
            "capability:multivariate",
            "capability:unequal_length",
            "capability:missing_values",
            "capability:train_estimate",
            "capability:feature_importance",
            "capability:contractable",
            "capability:categorical_in_X",
        ]

        TAGS_TO_SET = {
            "X_inner_mtype": MTYPE_LIST_PANEL,
            "y_inner_mtype": MTYPE_LIST_TABLE,
        }

        self.clone_tags(delegate, tag_names=TAGS_TO_DELEGATE)
        self.set_tags(**TAGS_TO_SET)

        return self

    def _fit(self, X, y):
        """Fit time series regressor to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb
        y : 1D np.array of float, of shape [n_instances] - regression labels for fitting
            indices correspond to instance indices in X

        Returns
        -------
        self : Reference to self.
        """
        estimator = self._get_delegate()
        estimator.fit(X=X, y=y)
        return self

    def _predict(self, X):
        """Predict labels for sequences in X.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        X : guaranteed to be of a type in self.get_tag("X_inner_mtype")
            if self.get_tag("X_inner_mtype") = "numpy3D":
                3D np.ndarray of shape = [n_instances, n_dimensions, series_length]
            if self.get_tag("X_inner_mtype") = "nested_univ":
                pd.DataFrame with each column a dimension, each cell a pd.Series
            for list of other mtypes, see datatypes.SCITYPE_REGISTER
            for specifications, see examples/AA_datatypes_and_datasets.ipynb

        Returns
        -------
        y : 1D np.array of float, of shape [n_instances] - predicted regression labels
            indices correspond to instance indices in X
        """
        estimator = self._get_delegate()
        return estimator.predict(X=X)

    def _get_fitted_params(self):
        """Get fitted parameters.

        private _get_fitted_params, called from get_fitted_params

        State required:
            Requires state to be "fitted".

        Returns
        -------
        fitted_params : dict with str keys
            fitted parameters, keyed by names of fitted parameter
        """
        estimator = self._get_delegate()
        return estimator.get_fitted_params()
