"""Monte Carlo Recursive Probabilistic Reduction Forecaster.

A reduction forecaster that wraps probabilistic tabular regressors (skpro)
to produce multi-step probabilistic forecasts using ancestral sampling.
"""

__author__ = ["marrov"]
__all__ = ["MCRecursiveProbaReductionForecaster"]

import inspect

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._base_proba import BaseProbaForecaster
from sktime.forecasting.compose._reduce import (
    _get_notna_idx,
    _ReducerMixin,
    slice_at_ix,
)
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.sklearn import prep_skl_df
from sktime.utils.warnings import warn


def _get_last_X_for_index(X, target_idx):
    """Get fallback exogenous values aligned to ``target_idx``.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame containing at least one row of exogenous data.
    target_idx : pd.Index or pd.MultiIndex
        Target index to align the fallback values to.

    Returns
    -------
    pd.DataFrame
        DataFrame with fallback values aligned to target_idx.
    """
    if len(X) == 0:
        raise ValueError("`X` must contain at least one row.")

    if isinstance(target_idx, pd.MultiIndex):
        if isinstance(X.index, pd.MultiIndex):
            levels = list(range(X.index.nlevels - 1))
            X_last = X.groupby(level=levels, as_index=False).tail(1).copy()
            X_last.index = X_last.index.droplevel(-1)
            target_no_time = target_idx.droplevel(-1)
            X_last = X_last.reindex(target_no_time)
            X_last.index = target_idx
            return X_last

        X_last_vals = np.tile(X.iloc[[-1]].to_numpy(), (len(target_idx), 1))
        X_last = pd.DataFrame(X_last_vals, columns=X.columns, index=target_idx)
        return X_last

    X_last = X.iloc[[-1]].copy()
    X_last.index = target_idx
    return X_last


def _align_X_columns(X, columns):
    """Align ``X`` columns to ``columns`` and remove duplicate labels.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to align columns for.
    columns : pd.Index or list
        Target column labels to align to.

    Returns
    -------
    pd.DataFrame
        DataFrame with aligned columns.
    """
    X = X.copy()
    columns = pd.Index(columns).drop_duplicates()

    # Use name-aware reindex to avoid silently reassigning columns by position.
    if X.columns.tolist() != list(columns):
        X = X.reindex(columns=columns)

    if X.columns.has_duplicates:
        X = X.loc[:, ~X.columns.duplicated(keep="first")]
        X = X.reindex(columns=columns)

    return X


def _pool_exogenous(X_hist, X_new=None):
    """Pool historic and new exogenous data with stable column schema.

    Parameters
    ----------
    X_hist : pd.DataFrame
        Historic exogenous data.
    X_new : pd.DataFrame, optional
        New exogenous data to pool with historic data.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame with deduplicated index (keep last).
    """
    x_cols = pd.Index(X_hist.columns).drop_duplicates()
    X_hist = _align_X_columns(X_hist, x_cols)

    if X_new is None:
        return X_hist

    X_new = _align_X_columns(X_new, x_cols)
    X_pool = pd.concat([X_hist, X_new], axis=0)
    X_pool = X_pool[~X_pool.index.duplicated(keep="last")]

    return X_pool


def _align_X_index(X, target_index):
    """Align exogenous row index to ``target_index`` by shape where possible.

    Parameters
    ----------
    X : pd.DataFrame
        DataFrame to align index for.
    target_index : pd.Index
        Target index to align to.

    Returns
    -------
    pd.DataFrame
        DataFrame with aligned index.
    """
    X = X.copy()
    if len(X) == len(target_index):
        X.index = target_index
        return X

    return X.reindex(target_index)


class MCRecursiveProbaReductionForecaster(BaseProbaForecaster, _ReducerMixin):
    """Monte Carlo Recursive reduction forecaster with probabilistic prediction.

    Implements recursive reduction with ancestral sampling for multi-step ahead
    probabilistic forecasting. Uses Monte Carlo sampling to generate multiple
    forecast trajectories, where each step is sampled from the predicted
    distribution conditioned on previously sampled values.

    This approach is inspired by DeepAR's ancestral sampling strategy, adapted
    for use with any tabular probabilistic regressor from `skpro`.

    Algorithm details:

    In ``fit``, given endogenous time series ``y`` and possibly exogenous ``X``:
        fits ``estimator`` to feature-label pairs for one-step-ahead prediction:
        features = ``y(t)``, ``y(t-1)``, ..., ``y(t-window_length+1)``,
                   if provided: ``X(t+1)``
        labels = ``y(t+1)``
        ranging over all ``t`` where the above have been observed

    In ``predict_proba``, given possibly exogenous ``X``, at cutoff time ``c``:
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

    Examples
    --------
    >>> from sklearn.linear_model import LinearRegression
    >>> from skpro.regression.residual import ResidualDouble
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.compose import MCRecursiveProbaReductionForecaster
    >>>
    >>> y = load_airline()
    >>>
    >>> base_estimator = LinearRegression()
    >>> estimator = ResidualDouble(base_estimator)
    >>> forecaster = MCRecursiveProbaReductionForecaster(estimator)
    >>>
    >>> forecaster.fit(y)  # doctest: +ELLIPSIS
    MCRecursiveProbaReductionForecaster(...)
    >>> y_pred_dist = forecaster.predict_proba(fh=range(1, 13))
    >>> len(y_pred_dist.mean()) == 12
    True
    """

    _tags = {
        "authors": ["marrov"],
        "python_dependencies": ["skpro>=2.11.0"],
        "requires-fh-in-fit": False,
        "capability:exogenous": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:random_state": True,
        "scitype:y": "both",
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "tests:libs": ["sktime.transformations.series.lag"],
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

        Follows the pattern from RecursiveReductionForecaster, using sktime
        transformers for lag feature creation.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to fit the forecaster to.
        X : pd.DataFrame, optional
            Exogenous time series.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        from sktime.transformations.series.lag import Lag

        impute_method = self.impute_method
        lags = self._lags

        lagger_y_to_X = Lag(lags=lags, index_out="extend")

        if impute_method is not None:
            if isinstance(impute_method, str):
                from sktime.transformations.series.impute import Imputer

                imputer = Imputer(method=impute_method)
            else:
                imputer = impute_method.clone()
            lagger_y_to_X = lagger_y_to_X * imputer

        self.lagger_y_to_X_ = lagger_y_to_X

        Xt = lagger_y_to_X.fit_transform(y)

        # Shift by 1 to align features at time t with target at time t+1
        lag_plus = Lag(lags=1, index_out="extend")
        Xtt = lag_plus.fit_transform(Xt)
        Xtt_notna_idx = _get_notna_idx(Xtt)
        notna_idx = Xtt_notna_idx.intersection(y.index)

        if len(notna_idx) == 0:
            warn(
                "No valid lagged rows were produced in fit; this can happen when "
                "`window_length` is larger than available observations. Falling back "
                "to constant mean forecasts via `y.mean()`.",
                obj=self,
                stacklevel=2,
            )
            self.estimator_ = y.mean()
            return self

        yt = y.loc[notna_idx]
        Xtt = Xtt.loc[notna_idx]

        if X is not None:
            Xtt = pd.concat([X.loc[notna_idx], Xtt], axis=1)

        Xtt = prep_skl_df(Xtt)
        yt = prep_skl_df(yt)

        estimator = clone(self.estimator)
        estimator.fit(Xtt, yt)
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
            Exogenous time series.

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions (mean of MC samples).
        """
        pred_dist = self._predict_proba(fh=fh, X=X, marginal=True)
        return pred_dist.mean()

    def _predict_proba(self, fh, X, marginal=True):
        """Compute probabilistic forecasts using Monte Carlo ancestral sampling.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X : pd.DataFrame, optional
            Exogenous time series.
        marginal : bool, optional, default=True
            Whether returned distribution is marginal by time index.

        Returns
        -------
        pred_dist : skpro BaseDistribution
            Predictive distribution.
            Returns an Empirical distribution from the MC samples.
        """
        if self._X is not None:
            X_pool = _pool_exogenous(self._X, X)
        else:
            X_pool = X

        fh_idx = self._get_expected_pred_idx(fh=fh)
        y_cols = self._y.columns

        # Fallback for edge case: no valid training data
        if isinstance(self.estimator_, pd.Series):
            n_horizons = len(fh_idx)
            samples = np.full(
                (self.n_samples, n_horizons, len(y_cols)),
                self.estimator_.values,
            )
            pred_dist = self._build_empirical_distribution(samples, fh_idx, y_cols)
            return pred_dist

        all_trajectories, fh_time_idx = self._generate_mc_trajectories(fh, X_pool)

        pred_dist = self._build_empirical_from_trajectories(
            all_trajectories, fh_idx, fh_time_idx, y_cols
        )

        return pred_dist

    def _generate_mc_trajectories(self, fh, X_pool):
        """Generate MC trajectories for recursive probabilistic prediction.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X_pool : pd.DataFrame or None
            Pooled exogenous data.

        Returns
        -------
        trajectories : dict
            Dictionary mapping instance identifiers to trajectory arrays.
            For single series, key is None.
            Each array has shape (n_samples, n_horizons, n_y_cols).
        fh_time_idx : pd.Index
            The time index for the forecasting horizon (without instance levels).
        """
        from sktime.transformations.series.lag import Lag

        n_samples = self.n_samples
        y_cols = self._y.columns

        rng = np.random.default_rng(self.random_state)

        fh_rel = fh.to_relative(self.cutoff)
        y_lags_rel = list(fh_rel)
        horizon_to_idx = {h: i for i, h in enumerate(y_lags_rel)}
        max_horizon = max(y_lags_rel)
        n_horizons = len(y_lags_rel)

        lagger_y_to_X = self.lagger_y_to_X_
        estimator = self.estimator_

        y_index = self._y.index
        is_hierarchical = isinstance(y_index, pd.MultiIndex)

        if is_hierarchical:
            instance_idx = y_index.droplevel(-1).unique()
        else:
            instance_idx = [None]

        n_y_cols = len(y_cols)
        trajectories = {
            inst: np.zeros((n_samples, n_horizons, n_y_cols)) for inst in instance_idx
        }

        # Build initial feature template from lagged data
        Xt_initial = lagger_y_to_X.transform(self._y)
        lag_plus_init = Lag(lags=1, index_out="extend")
        if isinstance(self.impute_method, str):
            from sktime.transformations.series.impute import Imputer

            lag_plus_init = lag_plus_init * Imputer(method=self.impute_method)
        Xtt_initial = lag_plus_init.fit_transform(Xt_initial)
        y_plus_one_init = lag_plus_init.fit_transform(self._y)

        first_predict_idx = y_plus_one_init.iloc[[-1]].index.get_level_values(-1)[0]
        Xtt_template = slice_at_ix(Xtt_initial, first_predict_idx)
        x_template_index = Xtt_template.index

        X_fallback_df = None
        exog_cols = None

        if X_pool is not None:
            exog_cols = pd.Index(X_pool.columns).drop_duplicates()
            X_fallback_df = _get_last_X_for_index(X_pool, x_template_index)
            X_fallback_df = _align_X_columns(X_fallback_df, exog_cols)
            try:
                X_at_idx = slice_at_ix(X_pool, first_predict_idx)
                if len(X_at_idx) == 0:
                    raise KeyError(first_predict_idx)
                X_at_idx = _align_X_index(X_at_idx, x_template_index)
                X_at_idx = _align_X_columns(X_at_idx, exog_cols)
                X_at_idx = X_at_idx.fillna(X_fallback_df)
            except (KeyError, IndexError):
                X_at_idx = X_fallback_df

            X_fallback_df = X_at_idx.copy()
            Xtt_template = pd.concat([X_at_idx, Xtt_template], axis=1)

        feature_cols = Xtt_template.columns.tolist()
        n_instances = len(Xtt_template) if is_hierarchical else 1

        # Initialize feature arrays: shape (n_samples, n_instances, n_features)
        Xtt_template = prep_skl_df(Xtt_template)
        initial_features = Xtt_template.values
        feature_cols = Xtt_template.columns
        sample_features = np.tile(initial_features, (n_samples, 1, 1))

        # Precompute exogenous features for each horizon step
        fh_abs = fh.to_absolute(self.cutoff)
        X_features_by_step = {}
        if X_pool is not None:
            for step_idx in range(max_horizon):
                horizon = step_idx + 1
                fh_step = ForecastingHorizon(
                    [horizon], is_relative=True, freq=self._cutoff
                )
                fh_step_abs = fh_step.to_absolute_index(self._cutoff)
                predict_time = fh_step_abs[0]
                try:
                    X_at_idx = slice_at_ix(X_pool, predict_time)
                    if len(X_at_idx) == 0:
                        raise KeyError(predict_time)
                    X_at_idx = _align_X_index(X_at_idx, x_template_index)
                    X_at_idx = _align_X_columns(X_at_idx, exog_cols)
                    if X_fallback_df is not None:
                        X_at_idx = X_at_idx.fillna(X_fallback_df)
                    X_features_by_step[step_idx] = prep_skl_df(X_at_idx).values
                    X_fallback_df = X_at_idx
                except (KeyError, IndexError):
                    if X_fallback_df is not None:
                        X_features_by_step[step_idx] = prep_skl_df(X_fallback_df).values

        n_lag_features = Xt_initial.shape[1]
        n_exog_features = len(exog_cols) if X_pool is not None else 0
        sample_supports_random_state = None

        for step_idx in range(max_horizon):
            batch_features = sample_features.reshape(n_samples * n_instances, -1)

            if step_idx in X_features_by_step:
                X_vals = X_features_by_step[step_idx]
                X_batch = np.tile(X_vals, (n_samples, 1))
                if n_exog_features > 0:
                    batch_features[:, :n_exog_features] = X_batch

            Xt_batch = pd.DataFrame(batch_features, columns=feature_cols)

            pred_dist = estimator.predict_proba(Xt_batch)

            if sample_supports_random_state is None:
                try:
                    sample_supports_random_state = (
                        "random_state" in inspect.signature(pred_dist.sample).parameters
                    )
                except (TypeError, ValueError):
                    sample_supports_random_state = False

            if self.random_state is None:
                sampled_df = pred_dist.sample(n_samples=1)
            elif sample_supports_random_state:
                sample_seed = int(rng.integers(0, 2**31))
                try:
                    sampled_df = pred_dist.sample(
                        n_samples=1,
                        random_state=sample_seed,
                    )
                except TypeError:
                    random_state_prev = np.random.get_state()
                    np.random.seed(sample_seed)
                    try:
                        sampled_df = pred_dist.sample(n_samples=1)
                    finally:
                        np.random.set_state(random_state_prev)
            else:
                sample_seed = int(rng.integers(0, 2**31))
                # Fallback for distributions that do not expose a random_state kwarg:
                # temporarily set NumPy global RNG to preserve reproducibility.
                random_state_prev = np.random.get_state()
                np.random.seed(sample_seed)
                try:
                    sampled_df = pred_dist.sample(n_samples=1)
                finally:
                    np.random.set_state(random_state_prev)

            sampled_values = (
                sampled_df.values.flatten()
                if hasattr(sampled_df, "values")
                else np.array(sampled_df).flatten()
            )
            sampled_values = sampled_values.reshape(n_samples, n_instances, n_y_cols)

            # Store trajectory values for requested horizons
            horizon = step_idx + 1
            traj_idx = horizon_to_idx.get(horizon)
            if traj_idx is not None:
                if is_hierarchical:
                    for inst_num, inst in enumerate(instance_idx):
                        trajectories[inst][:, traj_idx, :] = sampled_values[
                            :, inst_num, :
                        ]
                else:
                    trajectories[None][:, traj_idx, :] = sampled_values[:, 0, :]

            # Shift lag features and insert new sampled values at lag_0
            window_length = self.window_length
            start_idx = n_exog_features
            lag_end_idx = start_idx + n_lag_features

            # Shift lag block one step to the right and write new values into lag_0.
            if window_length > 1:
                sample_features[:, :, start_idx + n_y_cols : lag_end_idx] = (
                    sample_features[:, :, start_idx : lag_end_idx - n_y_cols]
                )

            sample_features[:, :, start_idx : start_idx + n_y_cols] = sampled_values

        # Get the time-only index for the forecast horizon
        fh_time_idx = pd.Index(list(fh_abs))

        return trajectories, fh_time_idx

    def _build_empirical_from_trajectories(
        self, trajectories, fh_idx, fh_time_idx, y_cols
    ):
        """Build Empirical distribution from trajectory dictionary.

        Parameters
        ----------
        trajectories : dict
            Dictionary mapping instance identifiers to trajectory arrays.
        fh_idx : pd.Index
            The full forecast index (may include instance levels).
        fh_time_idx : pd.Index
            The time-only forecast index.
        y_cols : pd.Index
            Column names for the target variable.

        Returns
        -------
        pred_dist : skpro Empirical
            The empirical distribution built from samples.
        """
        from skpro.distributions import Empirical

        n_samples = self.n_samples
        is_hierarchical = isinstance(fh_idx, pd.MultiIndex)

        n_horizons = len(fh_time_idx)
        n_rows = n_samples * n_horizons

        # Pre-build repeated index arrays (C-order so sample varies slowest)
        sample_arange = np.arange(n_samples)
        s_arr = np.repeat(sample_arange, n_horizons)
        fh_time_arr = np.array(list(fh_time_idx), dtype=object)
        t_arr = np.tile(fh_time_arr, n_samples)

        if is_hierarchical:
            fh_names = list(fh_idx.names)
            spl_names = ["sample"] + fh_names

            spl_parts = []
            for inst, traj_array in trajectories.items():
                # Build per-instance index entirely from numpy arrays (no Python loops)
                if isinstance(inst, tuple):
                    arrays = (
                        [s_arr]
                        + [np.full(n_rows, v, dtype=object) for v in inst]
                        + [t_arr]
                    )
                else:
                    arrays = [s_arr, np.full(n_rows, inst, dtype=object), t_arr]

                idx = pd.MultiIndex.from_arrays(arrays, names=spl_names)
                # traj_array: (n_samples, n_horizons, n_y_cols) — reshape is zero-copy
                vals = traj_array.reshape(n_rows, len(y_cols))
                spl_parts.append(pd.DataFrame(vals, index=idx, columns=y_cols))

            spl_df = pd.concat(spl_parts, axis=0)
        else:
            # Single series: (sample, time) index
            traj_array = trajectories[None]
            multi_idx = pd.MultiIndex.from_arrays(
                [s_arr, t_arr], names=["sample", "time"]
            )
            vals = traj_array.reshape(n_rows, len(y_cols))
            spl_df = pd.DataFrame(vals, index=multi_idx, columns=y_cols)

        # Create Empirical distribution
        pred_dist = Empirical(spl=spl_df, index=fh_idx, columns=y_cols)

        return pred_dist

    def _build_empirical_distribution(self, samples, fh_idx, y_cols):
        """Build Empirical distribution from numpy array of samples.

        Parameters
        ----------
        samples : np.ndarray
            Shape (n_samples, n_horizons, n_cols) array.
        fh_idx : pd.Index
            Forecast horizon index.
        y_cols : pd.Index
            Column names.

        Returns
        -------
        pred_dist : skpro Empirical
            The empirical distribution.
        """
        from skpro.distributions import Empirical

        n_samples = samples.shape[0]

        sample_indices = []
        time_indices = []
        values = []

        for sample_idx in range(n_samples):
            for h_idx, time_idx in enumerate(fh_idx):
                sample_indices.append(sample_idx)
                time_indices.append(time_idx)
                values.append(samples[sample_idx, h_idx, :])

        multi_idx = pd.MultiIndex.from_arrays(
            [sample_indices, time_indices], names=["sample", "time"]
        )
        spl_df = pd.DataFrame(np.vstack(values), index=multi_idx, columns=y_cols)

        return Empirical(spl=spl_df, index=fh_idx, columns=y_cols)

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        common = {"window_length": 3, "random_state": 42}
        params1 = {**common, "n_samples": 10, "pooling": "local"}
        params2 = {**common, "n_samples": 20, "pooling": "global"}

        if _check_soft_dependencies("skpro", severity="none"):
            from sklearn.linear_model import LinearRegression
            from skpro.regression.residual import ResidualDouble

            est = ResidualDouble(LinearRegression())

            params1 = {**params1, "estimator": est}
            params2 = {**params2, "estimator": est}

            params = [params1, params2]
        else:
            params = [{**params1, "estimator": "placeholder"}]

        return params
