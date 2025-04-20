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
        self.requires_residuals = self.reconciliation_method in [
            "wls_var",
            "mint_cov",
            "mint_shrink",
        ]
        self.agg_method = "mean"
        self.aggregation_levels = None
        self.forecasters = {}
        super().__init__()

        TRFORM_LIST = ["bu", "ols", "wls_str", "wls_var", "mint_cov", "mint_shrink"]
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

    def _thief_s_matrix(self, agg_factors):
        """Build full S matrix for THieF using all aggregation levels."""
        all_S = []
        agg_list = list(agg_factors)
        agg_list.sort(reverse=True)
        for f in agg_list:
            if agg_list[0] % f == 0:
                S_f = self._build_summing_matrix(agg_list[0], f)
                all_S.append(S_f)

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
        """Convert fh into aggregated levels."""
        from sktime.forecasting.base import ForecastingHorizon

        fh_period = fh.to_pandas()
        if not isinstance(fh_period, pd.PeriodIndex):
            fh_period = fh_period.to_period("M")

        reference_period = fh_period.min()
        fh_diffs = np.array([(p - reference_period).n for p in fh_period]) + 1
        new_vals = np.unique(
            [int(np.ceil(i / level)) for i in fh_diffs if (i / level) >= 1]
        )

        return ForecastingHorizon(new_vals, is_relative=True)

    def _reconcile_forecasts(self, forecasts):
        """Reconcile forecasts using the specified reconciliation method."""
        self._Y_base = np.vstack(
            [forecasts[level].values for level in forecasts.keys()]
        )
        self._S = self._thief_s_matrix(self.forecasters.keys())
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

            if self.requires_residuals:
                fh_insample = ForecastingHorizon(y_agg.index, is_relative=False)
                y_pred_in = forecaster.predict(fh=fh_insample)
                residuals = y_agg.squeeze() - y_pred_in.squeeze()
                self._residuals[level] = residuals

        return self

    def _predict(self, fh, X=None):
        """Predict using the fitted forecaster."""
        self._forecast_store = {}
        for level, forecaster in self.forecasters.items():
            fh_level = self._divide_fh(fh, level)

            if len(fh_level) > 0:
                y_pred_agg = forecaster.predict(fh=fh_level, X=X)
                if not y_pred_agg.isna().values.any():
                    self._forecast_store[level] = y_pred_agg
            else:
                raise AssertionError("Forecast horizon cannot be empty")

        result = self._reconcile_forecasts(self._forecast_store)

        return result

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
                "aggregation_levels": [1, 2, 4, 12],
                "reconciliation_method": "ols",
            },
            {
                "base_forecaster": NaiveForecaster(strategy="last"),
                "aggregation_levels": [1, 3, 6, 12],
                "reconciliation_method": "bu",
            },
            {
                "base_forecaster": NaiveForecaster(strategy="drift"),
                "aggregation_levels": [1, 2, 4, 8, 16],
                "reconciliation_method": "wls_str",
            },
            {
                "base_forecaster": NaiveForecaster(strategy="mean"),
                "aggregation_levels": [1, 2, 4],
                "reconciliation_method": "mint_cov",
            },
            {
                "base_forecaster": NaiveForecaster(strategy="last"),
                "aggregation_levels": [1, 2, 3, 6, 12],
                "reconciliation_method": "mint_shrink",
            },
            {
                "base_forecaster": NaiveForecaster(strategy="last"),
                "aggregation_levels": [1, 2, 3, 6, 12],
                "reconciliation_method": "wls_var",
            },
        ]

        return params
