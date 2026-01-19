"""Monte Carlo Recursive Probabilistic Reduction Forecaster.

A reduction forecaster that wraps probabilistic tabular regressors (skpro)
to produce multi-step probabilistic forecasts using ancestral sampling.
"""

__author__ = ["marrov"]
__all__ = ["MCRecursiveProbaReductionForecaster"]

import numpy as np
import pandas as pd
from sklearn.base import clone

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._base_proba import BaseProbaForecaster
from sktime.forecasting.compose._reduce import _get_notna_idx, _ReducerMixin
from sktime.utils.dependencies import _check_soft_dependencies
from sktime.utils.sklearn import prep_skl_df


def _slice_at_ix(df, ix):
    """Slice dataframe at index value, return row as DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to slice.
    ix : hashable
        Index value to slice at.

    Returns
    -------
    pd.DataFrame
        Single row (or rows for MultiIndex) matching the index value.
    """
    if isinstance(df.index, pd.MultiIndex):
        # For MultiIndex, get all rows where the last level matches ix
        mask = df.index.get_level_values(-1) == ix
        return df.loc[mask]
    else:
        return df.loc[[ix]]


class MCRecursiveProbaReductionForecaster(BaseProbaForecaster, _ReducerMixin):
    """Monte Carlo Recursive reduction forecaster with probabilistic prediction.

    Implements recursive reduction with ancestral sampling for multi-step ahead
    probabilistic forecasting. Uses Monte Carlo sampling to generate multiple
    forecast trajectories, where each step is sampled from the predicted
    distribution conditioned on previously sampled values.

    This approach is inspired by DeepAR's ancestral sampling strategy, adapted
    for use with any tabular probabilistic regressor from `skpro`.

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

    trajectories_ : dict or None
        The most recently generated MC sample trajectories from predict_proba.
        For single series: dict with key None, value shape (n_samples, n_horizons).
        For hierarchical: dict with instance tuple keys, each (n_samples, n_horizons).
        Available after calling predict or predict_proba.

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
    >>> forecaster.fit(y)
    >>> y_pred_dist = forecaster.predict_proba(fh=range(1, 13))
    """

    _tags = {
        "authors": ["marrov"],
        "python_dependencies": ["skpro>=2.11.0"],
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

        Follows the pattern from RecursiveReductionForecaster, using sktime
        transformers for lag feature creation.

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
        from sktime.transformations.series.lag import Lag

        # Invalidate any cached predictions from previous fit
        self._invalidate_cache()

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

        # Fallback for edge case: no valid training data
        if isinstance(self.estimator_, pd.Series):
            n_horizons = len(fh_idx)
            samples = np.full(
                (self.n_samples, n_horizons, len(y_cols)),
                self.estimator_.values,
            )
            self.trajectories_ = {None: samples[:, :, 0]}
            pred_dist = self._build_empirical_distribution(samples, fh_idx, y_cols)
            self._cache_pred_dist(pred_dist, fh=fh, X=X)
            return pred_dist

        all_trajectories, fh_time_idx = self._generate_mc_trajectories(fh, X_pool)
        self.trajectories_ = all_trajectories

        pred_dist = self._build_empirical_from_trajectories(
            all_trajectories, fh_idx, fh_time_idx, y_cols
        )
        self._cache_pred_dist(pred_dist, fh=fh, X=X)

        return pred_dist

    def _generate_mc_trajectories(self, fh, X_pool):
        """Generate MC trajectories for recursive probabilistic prediction.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon.
        X_pool : pd.DataFrame or None
            Pooled exogeneous data.

        Returns
        -------
        trajectories : dict
            Dictionary mapping instance identifiers to trajectory arrays.
            For single series, key is None.
            Each array has shape (n_samples, n_horizons).
        fh_time_idx : pd.Index
            The time index for the forecasting horizon (without instance levels).
        """
        from sktime.transformations.series.lag import Lag

        n_samples = self.n_samples
        y_cols = self._y.columns

        rng = np.random.default_rng(self.random_state)

        fh_rel = fh.to_relative(self.cutoff)
        y_lags_rel = list(fh_rel)
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

        trajectories = {
            inst: np.zeros((n_samples, n_horizons)) for inst in instance_idx
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
        Xtt_template = _slice_at_ix(Xtt_initial, first_predict_idx)

        if X_pool is not None:
            try:
                X_at_idx = _slice_at_ix(X_pool, first_predict_idx)
                Xtt_template = pd.concat([X_at_idx, Xtt_template], axis=1)
            except (KeyError, IndexError):
                pass

        feature_cols = Xtt_template.columns.tolist()
        n_instances = len(Xtt_template) if is_hierarchical else 1

        # Initialize feature arrays: shape (n_samples, n_instances, n_features)
        initial_features = prep_skl_df(Xtt_template).values
        sample_features = np.tile(initial_features, (n_samples, 1, 1))

        # Precompute exogenous features for each horizon step
        fh_abs = fh.to_absolute(self.cutoff)
        X_features_by_step = {}
        X_fallback = None
        if X_pool is not None:
            for step_idx in range(max_horizon):
                horizon = step_idx + 1
                fh_step = ForecastingHorizon(
                    [horizon], is_relative=True, freq=self._cutoff
                )
                fh_step_abs = fh_step.to_absolute_index(self._cutoff)
                predict_time = fh_step_abs[0]
                try:
                    X_at_idx = _slice_at_ix(X_pool, predict_time)
                    X_features_by_step[step_idx] = prep_skl_df(X_at_idx).values
                    if X_fallback is None:
                        X_fallback = X_features_by_step[step_idx]
                except (KeyError, IndexError):
                    if X_fallback is not None:
                        X_features_by_step[step_idx] = X_fallback

        n_lag_features = Xt_initial.shape[1]
        n_exog_features = (
            len(feature_cols) - n_lag_features if X_pool is not None else 0
        )
        n_y_cols = len(y_cols)

        for step_idx in range(max_horizon):
            batch_features = sample_features.reshape(n_samples * n_instances, -1)

            if step_idx in X_features_by_step:
                X_vals = X_features_by_step[step_idx]
                X_batch = np.tile(X_vals, (n_samples, 1))
                if n_exog_features > 0:
                    batch_features[:, :n_exog_features] = X_batch

            Xt_batch = pd.DataFrame(batch_features, columns=feature_cols)
            Xt_batch = prep_skl_df(Xt_batch)

            pred_dist = estimator.predict_proba(Xt_batch)

            # Seed per-step for reproducibility (skpro uses global random state)
            if self.random_state is not None:
                sample_seed = rng.integers(0, 2**31)
                np.random.seed(sample_seed)
            sampled_df = pred_dist.sample(n_samples=1)
            sampled_values = (
                sampled_df.values.flatten()
                if hasattr(sampled_df, "values")
                else np.array(sampled_df).flatten()
            )
            sampled_values = sampled_values.reshape(n_samples, n_instances, n_y_cols)

            # Store trajectory values for requested horizons
            horizon = step_idx + 1
            if horizon in y_lags_rel:
                traj_idx = y_lags_rel.index(horizon)
                if is_hierarchical:
                    for inst_num, inst in enumerate(instance_idx):
                        trajectories[inst][:, traj_idx] = sampled_values[:, inst_num, 0]
                else:
                    trajectories[None][:, traj_idx] = sampled_values[:, 0, 0]

            # Shift lag features and insert new sampled values at lag_0
            window_length = self.window_length
            start_idx = n_exog_features

            # Vectorized lag shifting: roll features and update lag_0 with new samples
            if window_length > 1:
                # Shift lag features: move later positions to earlier positions
                # lag_i+1 <- lag_i for i = window_length-2 down to 0
                for lag in range(window_length - 1, 0, -1):
                    for col in range(n_y_cols):
                        src_pos = start_idx + (lag - 1) * n_y_cols + col
                        dst_pos = start_idx + lag * n_y_cols + col
                        if dst_pos < start_idx + n_lag_features:
                            sample_features[:, :, dst_pos] = sample_features[
                                :, :, src_pos
                            ]

            # Set lag_0 to the new sampled values
            for col in range(n_y_cols):
                sample_features[:, :, start_idx + col] = sampled_values[:, :, col]

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

        sample_indices = []
        row_indices = []
        values = []

        if is_hierarchical:
            # For hierarchical data, iterate over instances and time
            # The fh_idx already contains the full (instance..., time) tuples
            for inst, traj_array in trajectories.items():
                for sample_idx in range(n_samples):
                    for h_idx, time_val in enumerate(fh_time_idx):
                        sample_indices.append(sample_idx)
                        if isinstance(inst, tuple):
                            row_indices.append(inst + (time_val,))
                        else:
                            row_indices.append((inst, time_val))
                        values.append([traj_array[sample_idx, h_idx]])

            # Create MultiIndex with sample as first level, then instance+time levels
            # Get the level names from fh_idx
            fh_names = list(fh_idx.names)
            spl_names = ["sample"] + fh_names

            # Build tuples with sample as first element
            spl_tuples = [
                (s,) + (r if isinstance(r, tuple) else (r,))
                for s, r in zip(sample_indices, row_indices)
            ]
            multi_idx = pd.MultiIndex.from_tuples(spl_tuples, names=spl_names)
        else:
            # For single series, simple (sample, time) structure
            traj_array = trajectories[None]
            for sample_idx in range(n_samples):
                for h_idx, time_val in enumerate(fh_time_idx):
                    sample_indices.append(sample_idx)
                    row_indices.append(time_val)
                    values.append([traj_array[sample_idx, h_idx]])

            multi_idx = pd.MultiIndex.from_arrays(
                [sample_indices, row_indices], names=["sample", "time"]
            )

        spl_df = pd.DataFrame(np.array(values), index=multi_idx, columns=y_cols)

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

        if X is not None:
            try:
                X_key = (X.shape, hash(X.values.tobytes()))
            except (AttributeError, TypeError):
                # If X cannot be hashed, disable caching by returning None
                X_key = None
        else:
            X_key = None

        return fh_key, X_key

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
            if isinstance(spl.index, pd.MultiIndex):
                # spl has MultiIndex (sample, time)
                sample_level = spl.index.get_level_values(0)
                time_level = spl.index.get_level_values(-1)  # Last level is time

                if isinstance(instance_idx, tuple):
                    # Multiple hierarchy levels
                    new_tuples = [
                        (s,) + instance_idx + (t,)
                        for s, t in zip(sample_level, time_level)
                    ]
                else:
                    # Single hierarchy level
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

                spl_time_name = spl.index.names[-1]  # Last name is time
                new_names = [spl_sample_name] + instance_names + [spl_time_name]

                new_spl_index = pd.MultiIndex.from_tuples(new_tuples, names=new_names)
            else:
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

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        common = {"window_length": 3}
        params1 = {**common, "n_samples": 10, "pooling": "local"}
        params2 = {**common, "n_samples": 20, "pooling": "global"}

        params = []

        if _check_soft_dependencies("skpro", severity="none"):
            from sklearn.linear_model import LinearRegression
            from skpro.regression.residual import ResidualDouble

            est = ResidualDouble(LinearRegression())

            params1 = {**params1, "estimator": est}
            params2 = {**params2, "estimator": est}

            params = [params1, params2]

        return params
