"""Direct Probabilistic Reduction Forecaster.

A custom reduction forecaster that extends DirectReductionForecaster to support
probabilistic predictions when the underlying estimator supports them (e.g., skpro
regressors like XGBoostLSS).
"""

__author__ = ["marrov"]
__all__ = ["DirectProbaReductionForecaster", "MCRecursiveProbaReductionForecaster"]

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


class MCRecursiveProbaReductionForecaster(BaseProbaForecaster, _ReducerMixin):
    """Monte Carlo Recursive reduction forecaster with probabilistic prediction.

    Implements recursive reduction with ancestral sampling for multi-step ahead
    probabilistic forecasting. Uses Monte Carlo sampling to generate multiple
    forecast trajectories, where each step is sampled from the predicted
    distribution conditioned on previously sampled values.

    This approach is inspired by DeepAR's ancestral sampling strategy, adapted
    for use with any tabular probabilistic regressor (e.g., XGBoostLSS from skpro).

    Algorithm details:

    In ``fit``, given endogeneous time series ``y`` and possibly exogeneous ``X``:
        fits ``estimator`` to feature-label pairs for one-step-ahead prediction:
        features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_length+1)``,
                   if provided: ``X(t+1)``
        labels = ``y(t+1)``
        ranging over all ``t`` where the above have been observed

    In ``predict_proba``, given possibly exogeneous ``X``, at cutoff time ``c``:
        1. Generate ``n_samples`` Monte Carlo trajectories using ancestral sampling
        2. For each trajectory and each horizon step ``h``:
           a. Get probabilistic prediction from estimator using lagged features
           b. Sample one value from the predicted distribution
           c. Use this sampled value as input for the next step
        3. Construct empirical distribution from the ``n_samples`` trajectories

    In ``predict``, returns the mean of the empirical distribution from MC samples.

    Parameters
    ----------
    estimator : skpro probabilistic regressor
        Tabular probabilistic regression algorithm used in reduction.
        Must support ``predict_proba`` method returning a distribution object.

    window_length : int, optional, default=10
        Window length used in the reduction algorithm (number of lags).

    n_samples : int, optional, default=100
        Number of Monte Carlo sample trajectories to generate.
        Higher values give more accurate distribution estimates but slower inference.

    impute_method : str, None, or sktime transformation, optional
        Imputation method to use for missing values in the lagged data.
        default="bfill"
        if str, admissible strings are of ``Imputer.method`` parameter.
        if sktime transformer, this transformer is applied to the lagged data.
        if None, no imputation is done.

    pooling : str, one of ["local", "global", "panel"], optional, default="local"
        Level on which data are pooled to fit the supervised regression model.
        "local" = unit/instance level, one reduced model per lowest hierarchy level
        "global" = top level, one reduced model overall, on pooled data
        "panel" = second lowest level, one reduced model per panel level (-2)

    random_state : int, RandomState instance or None, optional, default=None
        Controls the randomness of the Monte Carlo sampling.

    Attributes
    ----------
    estimator_ : fitted estimator
        The fitted probabilistic regressor.

    trajectories_ : np.ndarray or None
        The most recently generated MC sample trajectories from predict_proba.
        Shape is (n_samples, n_horizons, n_cols). Available after calling
        predict or predict_proba. Useful for analysis and visualization.

    Examples
    --------
    >>> from skpro.regression.xgboostlss import XGBoostLSS
    >>> from sktime.forecasting.compose import MCRecursiveProbaReductionForecaster
    >>> from sktime.datasets import load_airline
    >>>
    >>> y = load_airline()
    >>> estimator = XGBoostLSS(dist="Normal", n_trials=0)
    >>> forecaster = MCRecursiveProbaReductionForecaster(
    ...     estimator=estimator,
    ...     window_length=5,
    ...     n_samples=100,
    ... )
    >>> forecaster.fit(y, fh=[1, 2, 3])
    >>> y_pred_dist = forecaster.predict_proba(fh=[1, 2, 3])
    """

    _tags = {
        "authors": ["marrov"],
        "requires-fh-in-fit": False,
        "capability:exogenous": True,
        "capability:pred_int": True,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
    }

    def __init__(
        self,
        estimator,
        window_length=10,
        n_samples=100,
        impute_method="bfill",
        pooling="local",
        random_state=None,
    ):
        self.window_length = window_length
        self.estimator = estimator
        self.n_samples = n_samples
        self.impute_method = impute_method
        self.pooling = pooling
        self.random_state = random_state
        self._lags = list(range(window_length))
        super().__init__()

        # Initialize cache attributes for avoiding duplicate computation
        self._cached_pred_dist_ = None
        self._cached_fh_key_ = None
        self._cached_X_key_ = None
        # Public attribute for user access to MC trajectories
        self.trajectories_ = None

        # Detect if estimator is a probabilistic regressor (skpro)
        if hasattr(estimator, "get_tags"):
            _est_type = estimator.get_tag("object_type", "regressor", False)
        else:
            _est_type = "regressor"

        if _est_type != "regressor_proba":
            raise ValueError(
                "MCRecursiveProbaReductionForecaster requires an skpro probabilistic "
                f"regressor, but received estimator of type {_est_type}. "
                "Use RecursiveReductionForecaster for non-probabilistic estimators."
            )

        self._est_type = _est_type

        if pooling == "local":
            mtypes = "pd.DataFrame"
        elif pooling == "global":
            mtypes = ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"]
        elif pooling == "panel":
            mtypes = ["pd.DataFrame", "pd-multiindex"]
        else:
            raise ValueError(
                "pooling in MCRecursiveProbaReductionForecaster must be one of"
                ' "local", "global", "panel", '
                f"but found {pooling}"
            )
        self.set_tags(**{"X_inner_mtype": mtypes})
        self.set_tags(**{"y_inner_mtype": mtypes})

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to fit the forecaster to.
        X : pd.DataFrame, optional
            Exogeneous time series.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        from sktime.transformations.series.lag import Lag, ReducerTransform

        # Invalidate any cached predictions from previous fit
        self._invalidate_cache()

        impute_method = self.impute_method
        lags = self._lags

        # Create the lagger for transforming y into features
        lagger_y_to_X = ReducerTransform(lags=lags, impute_method=impute_method)
        self.lagger_y_to_X_ = lagger_y_to_X

        # Fit the lagger and transform y to get features
        Xt = lagger_y_to_X.fit_transform(X=y, y=X)

        # Create target by lagging y by 1 step
        lagger_y_to_y = Lag(lags=-1, index_out="original", keep_column_names=True)
        self.lagger_y_to_y_ = lagger_y_to_y

        yt = lagger_y_to_y.fit_transform(X=y)

        # Get valid indices (no NaN)
        Xt_notna_idx = _get_notna_idx(Xt)
        yt_notna_idx = _get_notna_idx(yt)
        notna_idx = Xt_notna_idx.intersection(yt_notna_idx)

        if len(notna_idx) == 0:
            self.empty_lags_ = True
            self.dummy_value_ = y.mean()
            return self
        else:
            self.empty_lags_ = False

        yt = yt.loc[notna_idx]
        Xt = Xt.loc[notna_idx]

        Xt = prep_skl_df(Xt)
        yt = prep_skl_df(yt)

        # Clone and fit the estimator
        estimator = clone(self.estimator)
        estimator.fit(Xt, yt)
        self.estimator_ = estimator

        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Returns the mean of the Monte Carlo sampled distributions.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame, optional
            Exogeneous time series.

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions (mean of MC samples).
        """
        # Use cached distribution if available and valid
        pred_dist = self._get_cached_pred_dist(fh=fh, X=X)
        if pred_dist is None:
            pred_dist = self._predict_proba(fh=fh, X=X, marginal=True)
        return pred_dist.mean()

    def _predict_proba(self, fh, X, marginal=True):
        """Compute probabilistic forecasts using Monte Carlo ancestral sampling.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame, optional
            Exogeneous time series.
        marginal : bool, optional, default=True
            Whether returned distribution is marginal by time index.

        Returns
        -------
        pred_dist : skpro BaseDistribution
            Predictive distribution.
            Returns an Empirical distribution from the MC samples.
        """
        from skpro.distributions import Empirical

        # Check cache first to avoid recomputation
        cached = self._get_cached_pred_dist(fh=fh, X=X)
        if cached is not None:
            return cached

        if X is not None and self._X is not None:
            X_pool = X.combine_first(self._X)
        elif X is None and self._X is not None:
            X_pool = self._X
        else:
            X_pool = X

        fh_idx = self._get_expected_pred_idx(fh=fh)
        y_cols = self._y.columns

        # Handle empty lags case
        if self.empty_lags_:
            samples = np.full(
                (self.n_samples, len(fh_idx), len(y_cols)),
                self.dummy_value_.values[0],
            )
            # Store trajectories for user access
            self.trajectories_ = samples
            pred_dist = Empirical(
                spl=pd.DataFrame(
                    samples.reshape(-1, len(y_cols)),
                    columns=y_cols,
                    index=pd.MultiIndex.from_product(
                        [range(self.n_samples), fh_idx], names=["sample", "time"]
                    ),
                ),
                index=fh_idx,
                columns=y_cols,
            )
            self._cache_pred_dist(pred_dist, fh=fh, X=X)
            return pred_dist

        # Generate MC sample trajectories using ancestral sampling
        all_trajectories = self._generate_mc_trajectories(fh, X_pool)

        # Store trajectories for user access
        self.trajectories_ = all_trajectories

        # Reshape for Empirical distribution
        sample_indices = []
        time_indices = []
        values = []

        for sample_idx in range(self.n_samples):
            for h_idx, time_idx in enumerate(fh_idx):
                sample_indices.append(sample_idx)
                time_indices.append(time_idx)
                values.append(all_trajectories[sample_idx, h_idx, :])

        # Create MultiIndex DataFrame for Empirical
        multi_idx = pd.MultiIndex.from_arrays(
            [sample_indices, time_indices], names=["sample", "time"]
        )
        spl_df = pd.DataFrame(np.vstack(values), index=multi_idx, columns=y_cols)

        # Create Empirical distribution (non-parametric, preserves actual samples)
        pred_dist = Empirical(spl=spl_df, index=fh_idx, columns=y_cols)

        # Cache the result
        self._cache_pred_dist(pred_dist, fh=fh, X=X)

        return pred_dist

    def _get_cache_key(self, fh, X):
        """Generate a cache key based on forecasting horizon and exogeneous data.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogeneous time series.

        Returns
        -------
        tuple : (fh_key, X_key) for cache comparison
        """
        # Create a hashable key from fh
        fh_key = tuple(fh.to_relative(self.cutoff)) if fh is not None else None

        # Create a key from X (use shape and hash of values if available)
        if X is not None:
            try:
                X_key = (X.shape, hash(X.values.tobytes()))
            except (AttributeError, TypeError):
                X_key = id(X)  # Fallback to object id
        else:
            X_key = None

        return fh_key, X_key

    def _get_cached_pred_dist(self, fh, X):
        """Return cached prediction distribution if cache is valid.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogeneous time series.

        Returns
        -------
        pred_dist or None : cached distribution if valid, None otherwise
        """
        if self._cached_pred_dist_ is None:
            return None

        fh_key, X_key = self._get_cache_key(fh, X)

        if self._cached_fh_key_ == fh_key and self._cached_X_key_ == X_key:
            return self._cached_pred_dist_

        return None

    def _cache_pred_dist(self, pred_dist, fh, X):
        """Cache the prediction distribution with its parameters.

        Parameters
        ----------
        pred_dist : skpro BaseDistribution
            The prediction distribution to cache.
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame or None
            Exogeneous time series.
        """
        fh_key, X_key = self._get_cache_key(fh, X)
        self._cached_pred_dist_ = pred_dist
        self._cached_fh_key_ = fh_key
        self._cached_X_key_ = X_key

    def _invalidate_cache(self):
        """Invalidate the prediction cache.

        Should be called when the model state changes (e.g., after fit or update).
        """
        self._cached_pred_dist_ = None
        self._cached_fh_key_ = None
        self._cached_X_key_ = None
        self.trajectories_ = None

    def _concat_distributions_hierarchical(self, dist_list, yvec):
        """Concatenate Empirical distributions from vectorized prediction.

        This override handles Empirical distributions properly by concatenating
        the sample DataFrames with proper hierarchical indexing.

        Parameters
        ----------
        dist_list : list of skpro Empirical
            List of Empirical distributions from each instance.
        yvec : VectorizedDF
            The vectorized data structure with instance information.

        Returns
        -------
        concat_dist : skpro Empirical
            Concatenated Empirical distribution with proper hierarchical index.
        """
        from skpro.distributions import Empirical

        if len(dist_list) == 0:
            raise ValueError("Cannot concatenate empty list of distributions")

        if len(dist_list) == 1:
            return dist_list[0]

        # Get the instance indices from the vectorized structure
        row_idx, _ = yvec.get_iter_indices()

        # Build concatenated sample DataFrames
        spl_dfs = []
        combined_indices = []

        for i, dist in enumerate(dist_list):
            # Get the sample DataFrame from this Empirical distribution
            spl = dist.spl  # DataFrame with MultiIndex (sample, time)

            # Get the instance identifier for this distribution
            instance_idx = row_idx[i]  # This is a tuple like ('h0_0', 'h1_0')

            # Create new MultiIndex with sample level first, then instance levels + time
            # Empirical requires: first index is sample, further indices are instance
            if isinstance(spl.index, pd.MultiIndex):
                # spl has MultiIndex (sample, time)
                sample_level = spl.index.get_level_values(0)
                time_level = spl.index.get_level_values(1)

                if isinstance(instance_idx, tuple):
                    # Multiple hierarchy levels
                    # New structure: (sample, h0, h1, time)
                    new_tuples = [
                        (s,) + instance_idx + (t,)
                        for s, t in zip(sample_level, time_level)
                    ]
                else:
                    # Single hierarchy level
                    # New structure: (sample, h0, time)
                    new_tuples = [
                        (s, instance_idx, t) for s, t in zip(sample_level, time_level)
                    ]

                # Get names: sample first, then instance names, then time
                spl_sample_name = spl.index.names[0]  # usually "sample"

                if isinstance(row_idx, pd.MultiIndex):
                    instance_names = list(row_idx.names)
                else:
                    instance_names = [
                        row_idx.name if hasattr(row_idx, "name") else "level_0"
                    ]

                spl_time_name = spl.index.names[1]  # usually "time"
                new_names = [spl_sample_name] + instance_names + [spl_time_name]

                new_spl_index = pd.MultiIndex.from_tuples(new_tuples, names=new_names)
            else:
                # spl has simple index (shouldn't happen for Empirical, but handle it)
                new_spl_index = spl.index

            # Create new DataFrame with updated index
            new_spl = spl.copy()
            new_spl.index = new_spl_index
            spl_dfs.append(new_spl)

            # Also update the distribution index (instance + time points)
            dist_index = dist.index
            if isinstance(instance_idx, tuple):
                new_dist_tuples = [instance_idx + (t,) for t in dist_index]
            else:
                new_dist_tuples = [(instance_idx, t) for t in dist_index]

            if isinstance(row_idx, pd.MultiIndex):
                instance_names = list(row_idx.names)
            else:
                instance_names = [
                    row_idx.name if hasattr(row_idx, "name") else "level_0"
                ]
            time_name = dist_index.name if dist_index.name is not None else "time"

            new_dist_index = pd.MultiIndex.from_tuples(
                new_dist_tuples,
                names=instance_names + [time_name],
            )
            combined_indices.append(new_dist_index)

        # Concatenate all sample DataFrames
        combined_spl = pd.concat(spl_dfs, axis=0)

        # Concatenate all distribution indices
        full_index = combined_indices[0]
        for idx in combined_indices[1:]:
            full_index = full_index.append(idx)

        # Get columns from first distribution
        columns = dist_list[0].columns

        # Create the combined Empirical distribution
        return Empirical(spl=combined_spl, index=full_index, columns=columns)

    def _fast_sample_from_dist(self, pred_dist, rng):
        """Sample from skpro distribution using fast numpy operations.

        This bypasses skpro's slow sample() method which creates DataFrames
        with MultiIndex. Instead, we extract distribution parameters and
        sample directly with numpy.

        Parameters
        ----------
        pred_dist : skpro BaseDistribution
            Distribution to sample from (one sample per row).
        rng : np.random.Generator
            Numpy random generator for reproducibility.

        Returns
        -------
        samples : np.ndarray
            Shape (n_rows, n_cols) array of samples.
        """
        dist_type = type(pred_dist).__name__

        # Try to extract parameters and sample with numpy
        # This is much faster than skpro's sample() for large batches
        try:
            if dist_type == "Normal":
                # Normal distribution: sample using loc (mu) and scale (sigma)
                mu = np.asarray(pred_dist.mu).flatten()
                sigma = np.asarray(pred_dist.sigma).flatten()
                samples = rng.normal(loc=mu, scale=sigma)
                return samples.reshape(-1, 1)

            elif dist_type == "LogNormal":
                # LogNormal: numpy uses different parameterization
                # skpro LogNormal has mu, sigma as log-space parameters
                mu = np.asarray(pred_dist.mu).flatten()
                sigma = np.asarray(pred_dist.sigma).flatten()
                samples = rng.lognormal(mean=mu, sigma=sigma)
                return samples.reshape(-1, 1)

            elif dist_type == "Laplace":
                mu = np.asarray(pred_dist.mu).flatten()
                scale = np.asarray(pred_dist.scale).flatten()
                samples = rng.laplace(loc=mu, scale=scale)
                return samples.reshape(-1, 1)

            elif dist_type == "Gamma":
                # Gamma: skpro uses alpha (shape) and beta (rate)
                # numpy uses shape and scale (1/rate)
                alpha = np.asarray(pred_dist.alpha).flatten()
                beta = np.asarray(pred_dist.beta).flatten()
                samples = rng.gamma(shape=alpha, scale=1.0 / beta)
                return samples.reshape(-1, 1)

            elif dist_type == "TDistribution":
                # T-distribution with location and scale
                df = np.asarray(pred_dist.df).flatten()
                mu = np.asarray(pred_dist.mu).flatten()
                sigma = np.asarray(pred_dist.sigma).flatten()
                # Sample standard t, then scale and shift
                samples = rng.standard_t(df=df) * sigma + mu
                return samples.reshape(-1, 1)

            elif dist_type == "Weibull":
                # Weibull: skpro uses scale and concentration (k/shape)
                scale = np.asarray(pred_dist.scale).flatten()
                k = np.asarray(pred_dist.k).flatten()
                samples = scale * rng.weibull(a=k)
                return samples.reshape(-1, 1)

            else:
                # Fallback to skpro's sample method for unknown distributions
                # This may be slow but ensures correctness
                sampled_values = pred_dist.sample(n_samples=1)
                if hasattr(sampled_values, "values"):
                    return sampled_values.values.reshape(-1, 1)
                else:
                    return np.array(sampled_values).reshape(-1, 1)

        except (AttributeError, TypeError):
            # If parameter extraction fails, fall back to skpro's sample
            sampled_values = pred_dist.sample(n_samples=1)
            if hasattr(sampled_values, "values"):
                return sampled_values.values.reshape(-1, 1)
            else:
                return np.array(sampled_values).reshape(-1, 1)

    def _generate_mc_trajectories(self, fh, X_pool):
        """Generate Monte Carlo sample trajectories using ancestral sampling.

        Optimized version that batches predictions across all samples at each
        horizon step, reducing the number of estimator calls from
        n_samples * max_horizon to just max_horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X_pool : pd.DataFrame or None
            Pooled exogeneous data.

        Returns
        -------
        trajectories : np.ndarray
            Shape (n_samples, n_horizons, n_cols) array of sampled trajectories.
        """
        y_cols = self._y.columns
        n_cols = len(y_cols)
        n_samples = self.n_samples
        window_length = self.window_length

        fh_rel = fh.to_relative(self.cutoff)
        fh_abs = fh.to_absolute(self.cutoff)
        y_lags_rel = list(fh_rel)

        # Get the maximum horizon to fill in gaps
        max_horizon = max(y_lags_rel)
        y_lags_no_gaps = list(range(1, max_horizon + 1))

        n_horizons = len(y_lags_rel)
        trajectories = np.zeros((n_samples, n_horizons, n_cols))

        # Set random state for reproducibility
        rng = np.random.default_rng(self.random_state)

        estimator = self.estimator_

        # Extract historical y values as numpy array for fast access
        # Shape: (T, n_cols) where T is the number of historical time points
        y_history = self._y.values  # (T, n_cols)

        # Initialize sample paths with the last window_length values from history
        # Shape: (n_samples, window_length, n_cols)
        # All samples start with the same historical window
        initial_window = y_history[-window_length:, :]  # (window_length, n_cols)
        # Replicate for all samples
        sample_windows = np.tile(initial_window, (n_samples, 1, 1))
        # sample_windows shape: (n_samples, window_length, n_cols)

        # Get exogenous features for each future time step if available
        X_features_by_step = {}
        if X_pool is not None:
            for step_idx, horizon in enumerate(y_lags_no_gaps):
                if step_idx < len(fh_abs):
                    predict_time = fh_abs[step_idx]
                    try:
                        X_at_idx = slice_at_ix(X_pool, predict_time)
                        X_features_by_step[step_idx] = X_at_idx.values.flatten()
                    except (KeyError, IndexError):
                        pass

        # Build feature column names to match training format
        # The lagger creates columns like "lag_0__col", "lag_1__col", etc.
        lag_col_names = []
        for lag in range(window_length):
            for col in y_cols:
                lag_col_names.append(f"lag_{lag}__{col}")

        # Add exogenous column names if present
        if X_pool is not None:
            X_col_names = list(X_pool.columns)
        else:
            X_col_names = []

        # Iterate over horizon steps (not samples!) - this is the key optimization
        for step_idx, horizon in enumerate(y_lags_no_gaps):
            # Build feature matrix for ALL samples at once
            # Features are the lagged y values from each sample's window
            # Shape: (n_samples, window_length * n_cols)
            lag_features = sample_windows.reshape(n_samples, -1)

            # Create DataFrame with proper column names
            Xt_batch = pd.DataFrame(lag_features, columns=lag_col_names)

            # Add exogenous features if available (same for all samples at this step)
            if step_idx in X_features_by_step:
                X_vals = X_features_by_step[step_idx]
                for i, col_name in enumerate(X_col_names):
                    Xt_batch[col_name] = X_vals[i]
                # Reorder columns to put X first (matching training format)
                Xt_batch = Xt_batch[X_col_names + lag_col_names]

            Xt_batch = prep_skl_df(Xt_batch)

            # Get probabilistic prediction for ALL samples at once
            pred_dist = estimator.predict_proba(Xt_batch)

            # Sample one value per row using fast numpy sampling
            # This bypasses skpro's slow DataFrame-based sample() method
            sampled_array = self._fast_sample_from_dist(pred_dist, rng)

            # Ensure correct shape (n_samples, n_cols)
            sampled_array = sampled_array.reshape(n_samples, n_cols)

            # Store if this horizon is in our requested fh
            if horizon in y_lags_rel:
                traj_idx = y_lags_rel.index(horizon)
                trajectories[:, traj_idx, :] = sampled_array

            # Update sample windows: roll and append new values
            # Shift window left by 1 and append new sampled values
            sample_windows = np.roll(sample_windows, -1, axis=1)
            sample_windows[:, -1, :] = sampled_array

        return trajectories

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        # Import a simple probabilistic regressor for testing
        # Note: This requires skpro to be installed
        try:
            from sklearn.linear_model import LinearRegression
            from skpro.regression.residual import ResidualDouble

            est = ResidualDouble(LinearRegression())
        except ImportError:
            # Fallback if skpro not available
            from sklearn.linear_model import LinearRegression

            est = LinearRegression()

        params1 = {
            "estimator": est,
            "window_length": 3,
            "n_samples": 10,
            "pooling": "local",
        }
        params2 = {
            "estimator": est,
            "window_length": 3,
            "n_samples": 20,
            "pooling": "global",
        }

        return [params1, params2]
