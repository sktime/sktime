"""THieF Forecaster implementation."""

__author__ = ["satvshr"]


import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon
from sktime.forecasting.mapa import MAPAForecaster

# todo: add any necessary imports here

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class THieFForecaster(BaseForecaster):
    """
    Temporal Hierarchical Forecasting (THieF) implementation for sktime.

    THieF fits multiple instances of a base forecaster at different temporal aggregation
    levels, then reconciles their forecasts to ensure consistency across time scales.

    Parameters
    ----------
    base_forecaster : sktime-compatible forecaster
        The forecasting model to be used at each aggregation level.
    aggregation_levels : list of int, default=None
        The levels at which the time series will be aggregated
        (e.g., [1, 2, 4, 12] for monthly data).
        If None, the levels are automatically determined based on the frequency of `y`.
    reconciliation_method : str, default="ols"
        The method used to reconcile forecasts across levels. Options include:
        - "ols": Ordinary Least Squares
        - "bu": Bottom-Up (uses the lowest-level forecast)
        - "mse": Variance scaling based on Mean Squared Error
        - "shr": Shrinkage estimator for covariance matrix
        - "sam": Sample covariance matrix
    """

    _tags = {
        "y_inner_mtype": ["pd.Series", "pd.DataFrame"],
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "authors": "satvshr",
    }

    def __init__(self, base_forecaster, reconciliation_method="ols"):
        self.base_forecaster = base_forecaster
        self.reconciliation_method = reconciliation_method
        # self.requires_residuals = self.reconciliation_method in [
        #     "wls_var",
        #     "mint_cov",
        #     "mint_shrink",
        # ]
        self.agg_method = "mean"
        self.aggregation_levels = None
        self.forecasters = {}
        super().__init__()

        TRFORM_LIST = ["bu", "ols", "wls_str"]
        if self.reconciliation_method not in TRFORM_LIST:
            raise ValueError(
                f"{self.reconciliation_method} has not been implemented for THieF."
            )

    def _build_summing_matrix(self, n_periods, agg_factor):
        """Build summing matrix S for given number of periods and aggregation factor."""
        k = n_periods // agg_factor
        S = np.zeros((k, n_periods))

        for i in range(k):
            end = n_periods - i * agg_factor
            start = end - agg_factor
            S[k - i - 1, start:end] = 1

        return S

    def _thief_s_matrix(self, forecasts):
        """Build full S matrix aligned with actual forecast output per level."""
        all_S = []
        # bottom level (most granular) is f=1
        base_len = len(forecasts[1])
        # iterate from highest aggregation down to 1
        for f in sorted(forecasts.keys(), reverse=True):
            n = len(forecasts[f])
            agg_factor = base_len // n
            S_full = self._build_summing_matrix(base_len, agg_factor)
            # trim to the first n rows so it matches forecasts[f]
            all_S.append(S_full[:n, :])
        return np.vstack(all_S)

    def _determine_aggregation_levels(self, y):
        """Determine the aggregation level based on the frequency of the time series."""
        m = None
        if hasattr(y.index, "freqstr") and y.index.freqstr:
            freq = y.index.freqstr
        elif isinstance(y.index, pd.PeriodIndex):
            idx = y.index.to_timestamp()
            freq = idx.freqstr
        elif isinstance(y.index, pd.RangeIndex):
            m = y.index.stop - y.index.start
        elif isinstance(y.index, pd.Index):
            m = y.index[-1] - y.index[0]

        freq_map = {"D": 7, "W": 52, "M": 12, "ME": 12, "H": 24, "Q": 4, "Y": 1}

        if m is None:
            if freq is None:
                raise ValueError(
                    "Could not infer frequency. Ensure timestamps are evenly spaced."
                )

            m = freq_map.get(freq[0].capitalize())

        if isinstance(m, pd.Timedelta):
            m = m.days

        if m is None:
            raise ValueError(f"Unsupported frequency '{freq}'.")

        aggregation_levels = [i for i in range(1, m + 1) if m % i == 0]

        return aggregation_levels

    def _divide_fh(self, fh, level):
        fh_period = fh.to_pandas()

        if pd.api.types.is_integer_dtype(fh_period.dtype):
            fh_diffs = fh_period.to_numpy()
        else:
            reference = fh_period.min()
            diffs = []
            for p in fh_period:
                delta = p - reference
                if isinstance(delta, pd.Timedelta):
                    diffs.append(delta.days)
                elif hasattr(delta, "n"):
                    diffs.append(delta.n)
                else:
                    diffs.append(int(delta))
            fh_diffs = np.array(diffs, dtype=int)

        new_vals = np.unique(
            [int(np.ceil(diff / level)) for diff in fh_diffs if diff >= level]
        )
        return ForecastingHorizon(new_vals, is_relative=True)

    def _reconcile_forecasts(self, forecasts):
        """Reconcile forecasts using the specified reconciliation method."""
        valid_levels = [f for f in forecasts.keys() if max(forecasts.keys()) % f == 0]
        # print(f"Valid aggregation levels: {valid_levels}")
        if not valid_levels:
            return np.empty(np.array([]))
        self._Y_base = np.vstack([forecasts[l].values for l in valid_levels])
        # print(f"Base Y: {self._Y_base}")
        filtered_forecasts = {l: forecasts[l] for l in valid_levels}
        self._S = self._thief_s_matrix(filtered_forecasts)
        # print(self.reconciliation_method)
        # if self.requires_residuals:
        #     self._resids = np.concatenate(
        #         [self._residuals[level].values for level in self._residuals.keys()]
        #     )

        if self.reconciliation_method == "bu":
            Y = forecasts[1].values  # Only bottom-level forecast is used
            return self._S @ Y

        elif self.reconciliation_method == "ols":
            G = np.linalg.pinv(self._S.T @ self._S) @ self._S.T
            return self._S @ G @ self._Y_base

        elif self.reconciliation_method == "wls_str":
            D = np.diag(self._S.sum(axis=1))
            G = np.linalg.pinv(self._S.T @ D @ self._S) @ self._S.T @ D
            return self._S @ G @ self._Y_base

        # elif self.reconciliation_method == "wls_var":
        #     variances = np.array([
        #     np.var(self._residuals[level]) for level in sorted(self._residuals)
        #     ])
        #     W = np.diag(variances)
        #     W_inv = np.linalg.pinv(W)
        #     G = np.linalg.pinv(self._S.T @ W_inv @ self._S) @ self._S.T @ W_inv
        #     return self._S @ G @ self._Y_base

        # elif self.reconciliation_method == "mint_cov":
        #     min_len = min(len(res) for res in self._residuals.values())
        #     aligned_resids = np.vstack([
        #         self._residuals[level][-min_len:] for level in sorted(self._residuals)
        #     ])
        #     W = np.cov(aligned_resids)
        #     W_inv = np.linalg.pinv(W)
        #     print(self._S.T.shape, W_inv.shape, self._S.shape)
        #     G = np.linalg.pinv(self._S.T @ W_inv @ self._S) @ self._S.T @ W_inv
        #     return self._S @ G @ self._Y_base

        # elif self.reconciliation_method == "mint_shrink":
        #     from sklearn.covariance import LedoitWolf

        #     min_len = min(len(res) for res in self._residuals.values())
        #     aligned_resids = np.vstack([
        #         self._residuals[level][-min_len:] for level in sorted(self._residuals)
        #     ])

        #     lw = LedoitWolf().fit(aligned_resids.T)
        #     W = lw.covariance_
        #     W_inv = np.linalg.pinv(W)
        #     G = np.linalg.pinv(self._S.T @ W_inv @ self._S) @ self._S.T @ W_inv
        #     return self._S @ G @ self._Y_base

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        if isinstance(y, pd.Series):
            y = y.to_frame()

        self._aggregation_levels = self._determine_aggregation_levels(y)
        self._residuals = {}
        for level in self._aggregation_levels:
            y_agg = MAPAForecaster._aggregate(self, y, level)

            if isinstance(y_agg.index, (pd.RangeIndex, pd.Index)):
                temp_index = pd.date_range(start="2000-01-01", periods=len(y))

                start_date = temp_index[0]
                original_freq = pd.infer_freq(temp_index)

                if original_freq is None:
                    raise ValueError(
                        "Could not infer frequency from generated DatetimeIndex."
                    )

                y_agg.index = pd.date_range(
                    start=start_date,
                    periods=len(y_agg),
                    freq=f"{level}{original_freq}",
                )

            forecaster = self.base_forecaster.clone()
            forecaster.fit(y_agg, X)
            self.forecasters[level] = forecaster

            # if self.requires_residuals:
            #     fh_insample = ForecastingHorizon(y_agg.index, is_relative=False)
            #     if len(y_agg) == 1:
            #         y_pred_in = y_agg.squeeze()
            #     else:
            #         y_pred_in = forecaster.predict(fh=fh_insample)
            #     residuals = y_agg.squeeze() - y_pred_in.squeeze()
            #     self._residuals[level] = residuals

        return self

    def _predict(self, fh, X=None):
        """Predict using the fitted forecaster."""
        self._forecast_store = {}
        # print(self.forecasters)
        for level, forecaster in self.forecasters.items():
            fh_level = self._divide_fh(fh, level)
            # print(fh_level)
            if len(fh_level) > 0:
                y_pred_agg = forecaster.predict(fh=fh_level, X=X)
                if not y_pred_agg.isna().values.any():
                    self._forecast_store[level] = y_pred_agg
            else:
                continue

        self._y_cols = (
            self._y.columns if isinstance(self._y, pd.DataFrame) else ["value"]
        )

        if not self._forecast_store:
            return pd.DataFrame(
                np.nan,
                index=fh.to_absolute(self.cutoff).to_pandas(),
                columns=self._y_cols,
            )
        # print(self._forecast_store)
        result = self._reconcile_forecasts(self._forecast_store)
        # print(result)
        n_needed = len(fh)
        n_available = result.shape[0]

        if n_available < n_needed:
            # pad with NaNs on top so that we have at least `n_needed` rows
            n_pad = n_needed - n_available
            pad_shape = (n_pad, result.shape[1])
            pad = np.full(pad_shape, np.nan)
            result = np.vstack([pad, result])

        # Now slice out exactly len(fh) rows
        final_forecast = result[-n_needed:, :]

        # 4) Build a DataFrame with the proper index and column names
        y_cols = self._y.columns if isinstance(self._y, pd.DataFrame) else ["value"]
        index = fh.to_absolute(self.cutoff).to_pandas()
        y_pred = pd.DataFrame(final_forecast, index=index, columns=y_cols)
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.naive import NaiveForecaster

        params = [
            {
                "base_forecaster": NaiveForecaster(strategy="mean"),
                "reconciliation_method": "ols",
            },
            {
                "base_forecaster": NaiveForecaster(strategy="last"),
                "reconciliation_method": "bu",
            },
            {
                "base_forecaster": NaiveForecaster(strategy="drift"),
                "reconciliation_method": "wls_str",
            },
            # {
            #     "base_forecaster": NaiveForecaster(strategy="mean"),
            #     "reconciliation_method": "mint_cov",
            # },
            # {
            #     "base_forecaster": NaiveForecaster(strategy="last"),
            #     "reconciliation_method": "mint_shrink",
            # },
            # {
            #     "base_forecaster": NaiveForecaster(strategy="last"),
            #     "reconciliation_method": "wls_var",
            # },
        ]

        return params
