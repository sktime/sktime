"""Direct Probabilistic Reduction Forecaster.

A custom reduction forecaster that extends DirectReductionForecaster to support
probabilistic predictions when the underlying estimator supports them (e.g., skpro
regressors like XGBoostLSS).
"""

__author__ = ["marrov"]
__all__ = ["DirectProbaReductionForecaster"]

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.multioutput import MultiOutputRegressor

from sktime.forecasting.base import BaseProbaForecaster
from sktime.forecasting.compose._reduce import _get_notna_idx, _ReducerMixin
from sktime.utils.sklearn import prep_skl_df


def slice_at_ix(df, ix):
    """Slice dataframe at index value, return row as DataFrame."""
    if isinstance(df.index, pd.MultiIndex):
        # For MultiIndex, we need to get all rows where the last level matches ix
        mask = df.index.get_level_values(-1) == ix
        return df.loc[mask]
    else:
        return df.loc[[ix]]


class DirectProbaReductionForecaster(BaseProbaForecaster, _ReducerMixin):
    """Direct reduction forecaster with probabilistic prediction capability.

    Extends the standard DirectReductionForecaster to support probabilistic
    predictions (predict_proba, predict_interval, predict_quantiles) when the
    underlying estimator is an skpro probabilistic regressor.

    This is useful when you want to use reduction-based forecasting with
    estimators like XGBoostLSS that can produce full predictive distributions.

    Algorithm details:

    In ``fit``, given endogeneous time series ``y`` and possibly exogeneous ``X``:
        fits ``estimator`` to feature-label pairs as defined as follows.
    if `X_treatment = "concurrent":
        features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_size)``, if provided:
        ``X(t+h)``
        labels = ``y(t+h)`` for ``h`` in the forecasting horizon
        ranging over all ``t`` where the above have been observed (are in the index)
        for each ``h`` in the forecasting horizon (separate estimator fitted per ``h``)
    if `X_treatment = "shifted":
        features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_size)``, if provided:
        ``X(t)``
        labels = ``y(t+h_1)``, ..., ``y(t+h_k)`` for ``h_j`` in the forecasting horizon
        ranging over all ``t`` where the above have been observed (are in the index)
        estimator is fitted as a multi-output estimator (for all ``h_j``
        simultaneously)

    In ``predict``, given possibly exogeneous ``X``, at cutoff time ``c``,
    if `X_treatment = "concurrent":
        applies fitted estimators' predict to
        feature = ``y(c)``, ``y(c-1)``, ..., ``y(c-window_size)``, if provided:
        ``X(c+h)``
        to obtain a prediction for ``y(c+h)``, for each ``h`` in the forecasting horizon
    if `X_treatment = "shifted":
        applies fitted estimator's predict to
        features = ``y(c)``, ``y(c-1)``, ..., ``y(c-window_size)``, if provided:
        ``X(c)``
        to obtain prediction for ``y(c+h_1)``, ..., ``y(c+h_k)`` for ``h_j`` in forec.
        horizon

    Parameters
    ----------
    estimator : sklearn regressor or skpro probabilistic regressor
        tabular regression algorithm used in reduction algorithm.
        If skpro regressor, resulting forecaster will have probabilistic capability.

    window_length : int, optional, default=10
        window length used in the reduction algorithm

    transformers : currently not used

    X_treatment : str, optional, one of "concurrent" (default) or "shifted"
        determines the timestamps of X from which y(t+h) is predicted, for horizon h
        "concurrent": y(t+h) is predicted from lagged y, and X(t+h), for all h in fh
            in particular, if no y-lags are specified, y(t+h) is predicted from X(t)
        "shifted": y(t+h) is predicted from lagged y, and X(t), for all h in fh
            in particular, if no y-lags are specified, y(t+h) is predicted from X(t+h)

    impute_method : str, None, or sktime transformation, optional
        Imputation method to use for missing values in the lagged data

        * default="bfill"
        * if str, admissible strings are of ``Imputer.method`` parameter, see there.
        * if sktime transformer, this transformer is applied to the lagged data.
        * if None, no imputation is done when applying ``Lag`` transformer

    pooling : str, one of ["local", "global", "panel"], optional, default="local"
        level on which data are pooled to fit the supervised regression model

    windows_identical : bool, optional, default=False
        Specifies whether all direct models use the same number of observations.
    """

    _tags = {
        "requires-fh-in-fit": True,
        "capability:exogenous": True,
        "capability:pred_int": True,  # Enable probabilistic predictions
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        transformers=None,
        X_treatment="concurrent",
        impute_method="bfill",
        pooling="local",
        windows_identical=False,
    ):
        self.window_length = window_length
        self.transformers = transformers
        self.transformers_ = None
        self.estimator = estimator
        self.X_treatment = X_treatment
        self.impute_method = impute_method
        self.pooling = pooling
        self.windows_identical = windows_identical
        self._lags = list(range(window_length))
        super().__init__()

        # Detect if estimator is a probabilistic regressor (skpro)
        if hasattr(estimator, "get_tags"):
            _est_type = estimator.get_tag("object_type", "regressor", False)
        else:
            _est_type = "regressor"

        # Store estimator type and set probabilistic capability accordingly
        self._est_type = _est_type
        self.set_tags(**{"capability:pred_int": _est_type == "regressor_proba"})

        if pooling == "local":
            mtypes = "pd.DataFrame"
        elif pooling == "global":
            mtypes = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        elif pooling == "panel":
            mtypes = ["pd.DataFrame", "pd-multiindex"]
        else:
            raise ValueError(
                "pooling in DirectProbaReductionForecaster must be one of"
                ' "local", "global", "panel", '
                f"but found {pooling}"
            )
        self.set_tags(**{"X_inner_mtype": mtypes})
        self.set_tags(**{"y_inner_mtype": mtypes})

    def _fit(self, y, X, fh):
        """Fit dispatcher based on X_treatment and windows_identical."""
        if (self.X_treatment == "shifted") and (self.windows_identical is True):
            return self._fit_multioutput(y=y, X=X, fh=fh)
        else:
            return self._fit_multiple(y=y, X=X, fh=fh)

    def _predict(self, fh=None, X=None):
        """Predict dispatcher based on X_treatment and windows_identical."""
        if self.X_treatment == "shifted":
            if self.windows_identical is True:
                return self._predict_multioutput(X=X, fh=fh)
            else:
                return self._predict_multiple(X=self._X, fh=fh)
        else:
            return self._predict_multiple(X=X, fh=fh)

    def _fit_multioutput(self, y, X=None, fh=None):
        """Fit to training data (multioutput case)."""
        from sktime.transformations.series.lag import Lag, ReducerTransform
        from sktime.utils.sklearn._tag_adapter import get_sklearn_tag

        impute_method = self.impute_method
        lags = self._lags
        trafos = self.transformers

        lagger_y_to_X = ReducerTransform(
            lags=lags, transformers=trafos, impute_method=impute_method
        )
        self.lagger_y_to_X_ = lagger_y_to_X

        fh_rel = fh.to_relative(self.cutoff)
        y_lags = list(fh_rel)
        y_lags = [-x for x in y_lags]
        lagger_y_to_y = Lag(lags=y_lags, index_out="original", keep_column_names=True)
        self.lagger_y_to_y_ = lagger_y_to_y

        yt = lagger_y_to_y.fit_transform(X=y)
        y_notna_idx = _get_notna_idx(yt)

        if len(y_notna_idx) == 0:
            self.empty_lags_ = True
            self.dummy_value_ = y.mean()
            return self
        else:
            self.empty_lags_ = False

        yt = yt.loc[y_notna_idx]

        Xt = lagger_y_to_X.fit_transform(X=y, y=X)
        Xt = Xt.loc[y_notna_idx]

        Xt = prep_skl_df(Xt)
        yt = prep_skl_df(yt)

        estimator = clone(self.estimator)

        # For non-probabilistic sklearn estimators, wrap in MultiOutputRegressor
        if self._est_type == "regressor":
            if not get_sklearn_tag(estimator, "capability:multioutput"):
                estimator = MultiOutputRegressor(estimator)
            yt_values = yt.values.flatten() if yt.shape[1] == 1 else yt
            estimator.fit(Xt, yt_values)
        else:
            # For skpro probabilistic regressors
            estimator.fit(Xt, yt)

        self.estimator_ = estimator

        return self

    def _predict_multioutput(self, fh=None, X=None):
        """Predict core logic (multioutput case)."""
        y_cols = self._y.columns
        fh_idx = self._get_expected_pred_idx(fh=fh)

        if self.empty_lags_:
            ret = pd.DataFrame(index=fh_idx, columns=y_cols)
            for i in ret.index:
                ret.loc[i] = self.dummy_value_
            return ret

        lagger_y_to_X = self.lagger_y_to_X_

        Xt = lagger_y_to_X.transform(X=self._y, y=self._X)
        Xt_lastrow = slice_at_ix(Xt, self.cutoff)
        Xt_lastrow = prep_skl_df(Xt_lastrow)

        estimator = self.estimator_
        y_pred = estimator.predict(Xt_lastrow)

        if self._est_type == "regressor":
            y_pred = y_pred.reshape((len(fh_idx), len(y_cols)))
            y_pred = pd.DataFrame(y_pred, columns=y_cols, index=fh_idx)
        else:
            # For skpro, predict returns a DataFrame directly
            if isinstance(y_pred, pd.DataFrame):
                y_pred.index = fh_idx

        if isinstance(y_pred.index, pd.MultiIndex):
            y_pred = y_pred.sort_index()

        return y_pred

    def _fit_multiple(self, y, X=None, fh=None):
        """Fit to training data (multiple estimators case)."""
        from sktime.transformations.series.lag import Lag, ReducerTransform

        impute_method = self.impute_method
        X_treatment = self.X_treatment
        windows_identical = self.windows_identical

        lags = self._lags

        fh_rel = fh.to_relative(self.cutoff)
        y_lags = list(fh_rel)
        y_lags = [-x for x in y_lags]

        lagger_y_to_y = dict()
        lagger_y_to_X = dict()
        self.lagger_y_to_y_ = lagger_y_to_y
        self.lagger_y_to_X_ = lagger_y_to_X

        self.estimators_ = []

        for lag in y_lags:
            t = Lag(lags=lag, index_out="original", keep_column_names=True)
            lagger_y_to_y[lag] = t

            yt = lagger_y_to_y[lag].fit_transform(X=y)

            impute_method = self.impute_method
            lags = self._lags
            trafos = self.transformers

            X_lag = lag if X_treatment == "concurrent" else 0

            lagger_y_to_X[lag] = ReducerTransform(
                lags=lags,
                shifted_vars_lag=X_lag,
                transformers=trafos,
                impute_method=impute_method,
            )

            Xtt = lagger_y_to_X[lag].fit_transform(X=y, y=X)
            Xtt_notna_idx = _get_notna_idx(Xtt)
            yt_notna_idx = _get_notna_idx(yt)
            notna_idx = Xtt_notna_idx.intersection(yt_notna_idx)

            yt = yt.loc[notna_idx]
            Xtt = Xtt.loc[notna_idx]

            if windows_identical:
                offset = np.abs(fh_rel.to_numpy()).max() - abs(lag)
                yt = yt[offset:]
                Xtt = Xtt[offset:]

            Xtt = prep_skl_df(Xtt)
            yt = prep_skl_df(yt)

            if len(notna_idx) == 0:
                self.estimators_.append(y.mean())
            else:
                estimator = clone(self.estimator)
                # For sklearn regressors, flatten the target
                if self._est_type == "regressor":
                    yt_values = yt.values.flatten() if yt.shape[1] == 1 else yt
                    estimator.fit(Xtt, yt_values)
                else:
                    # For skpro probabilistic regressors
                    estimator.fit(Xtt, yt)
                self.estimators_.append(estimator)

        return self

    def _predict_multiple(self, X=None, fh=None):
        """Predict core logic (multiple estimators case)."""
        from sktime.transformations.series.lag import Lag

        if X is not None and self._X is not None:
            X_pool = X.combine_first(self._X)
        elif X is None and self._X is not None:
            X_pool = self._X
        else:
            X_pool = X

        fh_idx = self._get_expected_pred_idx(fh=fh)
        y_cols = self._y.columns

        lagger_y_to_X = self.lagger_y_to_X_

        fh_rel = fh.to_relative(self.cutoff)
        fh_abs = fh.to_absolute(self.cutoff)
        y_lags = list(fh_rel)
        y_abs = list(fh_abs)

        y_pred_list = []

        for i, lag in enumerate(y_lags):
            predict_idx = y_abs[i]

            lag_plus = Lag(lag, index_out="extend", keep_column_names=True)

            Xt = lagger_y_to_X[-lag].transform(X=self._y, y=X_pool)
            Xtt = lag_plus.fit_transform(Xt)
            Xtt_predrow = slice_at_ix(Xtt, predict_idx)
            Xtt_predrow = prep_skl_df(Xtt_predrow)

            estimator = self.estimators_[i]

            if isinstance(estimator, pd.Series):
                y_pred_i = pd.DataFrame(index=[0], columns=y_cols)
                y_pred_i.iloc[0] = estimator
            else:
                y_pred_i = estimator.predict(Xtt_predrow)
            y_pred_list.append(y_pred_i)

        y_pred = np.concatenate(y_pred_list)
        y_pred = pd.DataFrame(y_pred, columns=y_cols, index=fh_idx)

        if isinstance(y_pred.index, pd.MultiIndex):
            y_pred = y_pred.sort_index()

        return y_pred

    def _predict_proba(self, fh, X, marginal=True):
        """Compute/return fully probabilistic forecasts.

        Parameters
        ----------
        fh : int, list, np.array or ForecastingHorizon (not optional)
            The forecasting horizon encoding the time stamps to forecast at.
        X : sktime time series object, optional (default=None)
            Exogeneous time series for the forecast
        marginal : bool, optional (default=True)
            whether returned distribution is marginal by time index

        Returns
        -------
        pred_dist : sktime BaseDistribution
            predictive distribution
        """
        if self._est_type != "regressor_proba":
            raise NotImplementedError(
                "predict_proba requires an skpro probabilistic regressor. "
                f"The estimator {type(self.estimator).__name__} is not probabilistic."
            )

        if self.X_treatment == "shifted" and self.windows_identical:
            return self._predict_proba_multioutput(X=X, fh=fh, marginal=marginal)
        else:
            return self._predict_proba_multiple(X=X, fh=fh, marginal=marginal)

    def _predict_proba_multioutput(self, fh=None, X=None, marginal=True):
        """Probabilistic prediction for multioutput case."""
        y_cols = self._y.columns
        fh_idx = self._get_expected_pred_idx(fh=fh)

        if self.empty_lags_:
            # Return a dummy distribution with the mean value
            from skpro.distributions import Normal

            ret = Normal(
                mu=self.dummy_value_.values[0],
                sigma=1.0,
                index=fh_idx,
                columns=y_cols,
            )
            return ret

        lagger_y_to_X = self.lagger_y_to_X_

        Xt = lagger_y_to_X.transform(X=self._y, y=self._X)
        Xt_lastrow = slice_at_ix(Xt, self.cutoff)
        Xt_lastrow = prep_skl_df(Xt_lastrow)

        estimator = self.estimator_
        # predict_proba returns a distribution object
        y_pred_dist = estimator.predict_proba(Xt_lastrow)

        # Update index to match forecasting horizon
        # The distribution returned by skpro has methods to update index
        if hasattr(y_pred_dist, "index"):
            # Create a new distribution with correct index
            # This depends on the specific distribution type
            y_pred_dist = self._reindex_distribution(y_pred_dist, fh_idx, y_cols)

        return y_pred_dist

    def _predict_proba_multiple(self, X=None, fh=None, marginal=True):
        """Probabilistic prediction for multiple estimators case."""
        from sktime.transformations.series.lag import Lag

        if X is not None and self._X is not None:
            X_pool = X.combine_first(self._X)
        elif X is None and self._X is not None:
            X_pool = self._X
        else:
            X_pool = X

        fh_idx = self._get_expected_pred_idx(fh=fh)
        y_cols = self._y.columns

        lagger_y_to_X = self.lagger_y_to_X_

        fh_rel = fh.to_relative(self.cutoff)
        fh_abs = fh.to_absolute(self.cutoff)
        y_lags = list(fh_rel)
        y_abs = list(fh_abs)

        # Collect distributions for each horizon step
        dist_list = []

        for i, lag in enumerate(y_lags):
            predict_idx = y_abs[i]

            lag_plus = Lag(lag, index_out="extend", keep_column_names=True)

            Xt = lagger_y_to_X[-lag].transform(X=self._y, y=X_pool)
            Xtt = lag_plus.fit_transform(Xt)
            Xtt_predrow = slice_at_ix(Xtt, predict_idx)
            Xtt_predrow = prep_skl_df(Xtt_predrow)

            estimator = self.estimators_[i]

            # Get probabilistic prediction from estimator
            dist_i = estimator.predict_proba(Xtt_predrow)
            # Reindex to match the prediction index
            dist_i = self._reindex_distribution(dist_i, pd.Index([predict_idx]), y_cols)

            dist_list.append(dist_i)

        # Concatenate distributions along the time axis
        pred_dist = self._concat_distributions(dist_list, fh_idx, y_cols)

        return pred_dist

    def _reindex_distribution(self, dist, new_index, columns):
        """Reindex a distribution object to have a new index.

        Parameters
        ----------
        dist : skpro BaseDistribution
            The distribution to reindex
        new_index : pd.Index
            The new index to use
        columns : pd.Index
            The columns to use

        Returns
        -------
        reindexed_dist : skpro BaseDistribution
            The distribution with updated index
        """
        # Get the distribution class
        dist_class = type(dist)

        # Get parameter names from the distribution's signature
        import inspect

        sig = inspect.signature(dist_class.__init__)
        param_names = [
            p
            for p in sig.parameters.keys()
            if p not in ["self", "index", "columns", "args", "kwargs"]
        ]

        # Extract parameters that exist as attributes on the distribution
        params = {}
        for param in param_names:
            if hasattr(dist, param):
                val = getattr(dist, param)
                if val is not None:
                    params[param] = val

        # Create new distribution with updated index
        if params:
            return dist_class(**params, index=new_index, columns=columns)
        else:
            # Fallback: return original distribution
            return dist

    def _concat_distributions(self, dist_list, new_index, columns):
        """Concatenate a list of distributions into a single distribution.

        Parameters
        ----------
        dist_list : list of skpro BaseDistribution
            List of distributions to concatenate
        new_index : pd.Index
            The full index for the concatenated distribution
        columns : pd.Index
            The columns to use

        Returns
        -------
        concat_dist : skpro BaseDistribution
            The concatenated distribution
        """
        if len(dist_list) == 0:
            raise ValueError("Cannot concatenate empty list of distributions")

        import inspect

        # Get the distribution class from the first non-None distribution
        dist_class = type(dist_list[0])

        # Get parameter names from the distribution's signature
        sig = inspect.signature(dist_class.__init__)
        param_names = [
            p
            for p in sig.parameters.keys()
            if p not in ["self", "index", "columns", "args", "kwargs"]
        ]

        # Collect parameters from all distributions and stack them
        param_arrays = {}

        for param in param_names:
            vals = []
            has_param = False
            for dist in dist_list:
                if hasattr(dist, param):
                    val = getattr(dist, param)
                    if val is not None:
                        has_param = True
                        vals.append(val)
            if has_param and len(vals) == len(dist_list):
                # Stack values
                stacked = np.vstack(vals)
                param_arrays[param] = stacked

        if param_arrays:
            return dist_class(**param_arrays, index=new_index, columns=columns)
        else:
            # Fallback: return first distribution
            return dist_list[0]

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        from sklearn.linear_model import LinearRegression

        est = LinearRegression()
        params1 = {
            "estimator": est,
            "window_length": 3,
            "X_treatment": "shifted",
            "pooling": "global",
            "windows_identical": True,
        }
        params2 = {
            "estimator": est,
            "window_length": 3,
            "X_treatment": "concurrent",
            "pooling": "global",
            "windows_identical": False,
        }

        return [params1, params2]
