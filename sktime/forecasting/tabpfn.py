# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements TabPFN forecaster."""

__author__ = ["priyanshuharshbodhi1"]
__all__ = ["TabPFNForecaster"]

from copy import deepcopy

import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


def _normalise_for_key(value):
    """Normalise mutable objects to hashable values for multiton cache keys."""
    if isinstance(value, dict):
        return tuple((k, _normalise_for_key(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_normalise_for_key(v) for v in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return repr(value)


class TabPFNForecaster(BaseForecaster):
    """Interface to TabPFN-TS for zero-shot time series forecasting.

    Wraps ``tabpfn_time_series.TabPFNTSPipeline`` as an sktime forecaster.
    Uses ``_multiton`` caching so multiple instances with the same config
    share one pipeline in memory.

    Parameters
    ----------
    max_context_length : int, default=4096
        Maximum number of context observations passed to the pipeline.
    tabpfn_mode : {"local", "client"}, default="local"
        ``"local"`` runs inference locally, ``"client"`` uses the
        hosted TabPFN cloud API.
    tabpfn_output_selection : {"mean", "median", "mode"}, default="median"
        Aggregation for the TabPFN ensemble output.
    tabpfn_model_config : dict or None, default=None
        Override keys merged into ``TABPFN_DEFAULT_CONFIG``.
        ``None`` keeps the package defaults.
    ignore_future_covariates_if_missing : bool, default=False
        If ``True``, missing future covariates at predict time are
        silently ignored instead of raising.

    References
    ----------
    .. [1] https://github.com/PriorLabs/tabpfn-time-series
    .. [2] Hoo, L.S.B. et al., "TabPFN-TS: Zero-shot Time Series
       Forecasting with TabPFNv2", arXiv preprint arXiv:2501.02945, 2025.

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.tabpfn import TabPFNForecaster
    >>> y = load_airline()
    >>> forecaster = TabPFNForecaster(tabpfn_mode="local")  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    TabPFNForecaster(...)
    >>> y_pred = forecaster.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        # contribution and dependency tags
        # ---------------------------------
        "authors": ["priyanshuharshbodhi1"],
        "maintainers": ["priyanshuharshbodhi1"],
        "python_dependencies": ["tabpfn-time-series"],
        # estimator type tags
        # --------------------
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
    }

    def __init__(
        self,
        max_context_length=4096,
        tabpfn_mode="local",
        tabpfn_output_selection="median",
        tabpfn_model_config=None,
        ignore_future_covariates_if_missing=False,
    ):
        self.max_context_length = max_context_length
        self.tabpfn_mode = tabpfn_mode
        self.tabpfn_output_selection = tabpfn_output_selection
        self.tabpfn_model_config = tabpfn_model_config
        self.ignore_future_covariates_if_missing = ignore_future_covariates_if_missing

        super().__init__()

    @staticmethod
    def _is_integer_index(index):
        return isinstance(index, pd.Index) and pd.api.types.is_integer_dtype(index)

    @classmethod
    def _require_supported_index(cls, index, var_name):
        is_ok = isinstance(
            index, (pd.DatetimeIndex, pd.PeriodIndex, pd.RangeIndex)
        ) or cls._is_integer_index(index)

        if not is_ok:
            raise NotImplementedError(
                f"{var_name} must have DatetimeIndex, PeriodIndex, RangeIndex, "
                f"or integer Index, but found {type(index)}."
            )

    @classmethod
    def _to_timestamp_index(cls, index, var_name):
        """Convert a supported index to DatetimeIndex for TabPFN-TS."""
        cls._require_supported_index(index, var_name=var_name)

        if isinstance(index, pd.PeriodIndex):
            return index.to_timestamp()
        if isinstance(index, pd.DatetimeIndex):
            return index

        # RangeIndex / plain integer -> synthetic daily timestamps
        return pd.DatetimeIndex(
            pd.to_datetime(index.to_numpy(), unit="D", origin="unix"),
            name=index.name,
        )

    def _get_tabpfn_kwargs(self, default_config, tabpfn_mode_enum):
        """Build kwargs dict for TabPFNTSPipeline construction."""
        mode_map = {
            "local": tabpfn_mode_enum.LOCAL,
            "client": tabpfn_mode_enum.CLIENT,
        }

        model_config = deepcopy(default_config)
        if self.tabpfn_model_config is not None:
            model_config.update(deepcopy(self.tabpfn_model_config))

        return {
            "max_context_length": self.max_context_length,
            "tabpfn_mode": mode_map[self.tabpfn_mode],
            "tabpfn_output_selection": self.tabpfn_output_selection,
            "tabpfn_model_config": model_config,
        }

    def _get_unique_tabpfn_key(self, tabpfn_kwargs):
        """Deterministic hashable key for the multiton cache."""
        key_dict = {k: _normalise_for_key(v) for k, v in tabpfn_kwargs.items()}
        return str(sorted(key_dict.items()))

    def _make_context_df(self):
        """Build the flat context DataFrame expected by predict_df."""
        context_df = self._y_context.rename("target").to_frame()
        if self._X_context is not None:
            context_df = context_df.join(self._X_context, how="left")

        context_df.index = self._to_timestamp_index(context_df.index, "y")
        context_df.index.name = "timestamp"
        context_df = context_df.reset_index()
        context_df.insert(0, "item_id", self._item_id)

        return context_df

    def _make_future_df(self, fh_abs, X):
        """Build the flat future DataFrame expected by predict_df."""
        fh_datetime = self._to_timestamp_index(fh_abs, "fh")
        future_df = pd.DataFrame({"item_id": self._item_id, "timestamp": fh_datetime})

        if X is None:
            if self._X_columns and not self.ignore_future_covariates_if_missing:
                raise ValueError(
                    "Future covariates are required because fit() received "
                    "exogenous variables. Either provide X in predict() or "
                    "set ignore_future_covariates_if_missing=True."
                )
            return future_df, fh_datetime

        self._require_supported_index(X.index, "X")
        X_future = X.copy()
        X_future.index = self._to_timestamp_index(X_future.index, "X")
        X_future = X_future.reindex(fh_datetime)
        future_df = pd.concat(
            [future_df.reset_index(drop=True), X_future.reset_index(drop=True)],
            axis=1,
        )

        return future_df, fh_datetime

    @staticmethod
    def _find_quantile_column(preds_df, alpha):
        """Find the column in preds_df matching quantile alpha."""
        if alpha in preds_df.columns:
            return alpha

        alpha_str = str(alpha)
        if alpha_str in preds_df.columns:
            return alpha_str

        for col in preds_df.columns:
            try:
                if float(col) == float(alpha):
                    return col
            except (TypeError, ValueError):
                continue

        raise KeyError(f"Quantile column for alpha={alpha} not found in predictions.")

    def _predict_tabpfn(self, fh, X, quantiles):
        """Call predict_df and align the output index."""
        fh_abs = fh.to_absolute_index(self.cutoff)
        self._require_supported_index(fh_abs, "fh")

        context_df = self._make_context_df()
        future_df, fh_datetime = self._make_future_df(fh_abs, X)

        preds_df = self._pipeline.predict_df(
            context_df=context_df,
            future_df=future_df,
            quantiles=[float(q) for q in quantiles],
        )

        # predict_df returns MultiIndex (item_id, timestamp)
        if isinstance(preds_df.index, pd.MultiIndex):
            if "item_id" in preds_df.index.names:
                preds_df = preds_df.xs(self._item_id, level="item_id", drop_level=True)
            else:
                preds_df = preds_df.droplevel(0)

        preds_df = preds_df.reindex(fh_datetime)
        preds_df.index = fh_abs

        return preds_df

    def _fit(self, y, X=None, fh=None):
        """Store training context and load the TabPFN pipeline."""
        from tabpfn_time_series import TABPFN_DEFAULT_CONFIG, TabPFNMode

        self._require_supported_index(y.index, "y")
        if X is not None:
            self._require_supported_index(X.index, "X")

        tabpfn_kwargs = self._get_tabpfn_kwargs(
            default_config=TABPFN_DEFAULT_CONFIG,
            tabpfn_mode_enum=TabPFNMode,
        )

        self._pipeline = _CachedTabPFN(
            key=self._get_unique_tabpfn_key(tabpfn_kwargs),
            tabpfn_kwargs=tabpfn_kwargs,
        ).load_from_checkpoint()

        self._y_context = y.copy()
        self._X_context = X.copy() if X is not None else None
        self._X_columns = [] if X is None else list(X.columns)
        self._item_id = "__sktime_series_0"

        return self

    def _predict(self, fh, X=None):
        """Forecast time series at future horizon."""
        preds_df = self._predict_tabpfn(fh, X, quantiles=[0.5])

        y_pred = preds_df["target"].copy()
        y_pred.name = self._y_context.name

        return y_pred

    def _predict_quantiles(self, fh, X, alpha):
        """Compute quantile forecasts for the given horizon."""
        preds_df = self._predict_tabpfn(fh, X, quantiles=alpha)

        var_names = self._get_varnames()
        col_index = pd.MultiIndex.from_product([var_names, alpha])
        pred_quantiles = pd.DataFrame(index=preds_df.index, columns=col_index)

        for a in alpha:
            col = self._find_quantile_column(preds_df, a)
            pred_quantiles[(var_names[0], a)] = preds_df[col].values

        return pred_quantiles

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {
                "max_context_length": 512,
                "tabpfn_mode": "local",
                "tabpfn_output_selection": "median",
            },
            {
                "max_context_length": 1024,
                "tabpfn_mode": "local",
                "tabpfn_output_selection": "mean",
                "ignore_future_covariates_if_missing": True,
            },
        ]


@_multiton
class _CachedTabPFN:
    """Multiton wrapper so identical configs share one pipeline in memory."""

    def __init__(self, key, tabpfn_kwargs):
        self.key = key
        self.tabpfn_kwargs = tabpfn_kwargs
        self._pipeline = None

    def load_from_checkpoint(self):
        """Load the TabPFNTSPipeline on first call, return cached after."""
        if self._pipeline is not None:
            return self._pipeline

        from tabpfn_time_series import TabPFNTSPipeline

        self._pipeline = TabPFNTSPipeline(**self.tabpfn_kwargs)
        return self._pipeline
