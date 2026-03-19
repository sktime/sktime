"""Implements TabPFN-TS forecaster."""

__author__ = ["liam-sbhoo", "Infonioknight"]

# Required_deps = [autogluon-timeseries, backoff, datasets, fev, gluonts, pandas]
# [python-dotenv, pyyaml, statsmodels, tabpfn, tabpfn-client, tabpfn-common-utils, tqdm]

__all__ = ["TabPFNTSForecaster"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton


class TabPFNTSForecaster(BaseForecaster):
    """
    Interface to the TabPFN-TS zero-shot time series forecaster.

    TabPFN-TS is a time series forecasting model built on top of the
    TabPFN family of tabular foundation models. It supports both
    univariate forecasting and forecasting with exogenous covariates.

    This class wraps the ``TabPFNTSPipeline`` from the
    ``tabpfn-time-series`` package and exposes it through the
    ``sktime`` forecasting interface.

    Parameters
    ----------
    tabpfn_mode : str, default="CLIENT"
        Execution mode for TabPFN inference.

        * "CLIENT" - use the hosted TabPFN inference service
        * "LOCAL" - run inference locally
    max_context_length : int, default=4096
        Maximum number of historical timesteps used as context.
        Longer contexts may improve accuracy at the cost of compute.

    tabpfn_output_selection : str, default="median"
        Method to aggregate TabPFN ensemble predictions.
        Supported values are ``"mean"``, ``"median"``, and ``"mode"``.

    References
    ----------
    .. [1] https://github.com/PriorLabs/tabpfn-time-series
    .. [2] Shi Bin Hoo, Samuel Müller, David Salinas, Frank Hutter
    The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting
    Models Based on Simple Features

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.tabpfn_ts import TabPFNTSForecaster
    >>> y = load_airline()
    >>> fh = ForecastingHorizon([1,2,3], is_relative=True)
    >>> f = TabPFNTSForecaster(tabpfn_mode="LOCAL")
    >>> _ = f.fit(y)
    >>> y_pred = f.predict(fh)
    """

    _tags = {
        "authors": ["liam-sbhoo", "Infonioknight"],
        # liam-sbhoo is from Prior Labs
        "maintainers": ["Infonioknight"],
        "python_dependencies": ["tabpfn_time_series"],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "requires-fh-in-fit": False,
        "capability:exogenous": True,
        "capability:insample": False,
        "capability:global_forecasting": False,
    }

    def __init__(
        self,
        tabpfn_mode="CLIENT",
        max_context_length=4096,
        tabpfn_output_selection="median",
    ):
        self.tabpfn_mode = tabpfn_mode
        self.max_context_length = max_context_length
        self.tabpfn_output_selection = tabpfn_output_selection
        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Initializes the underlying TabPFN-TS pipeline. Since TabPFN-TS is a
        zero-shot model, no parameter fitting is performed.

        Parameters
        ----------
        y : pd.DataFrame
            Target time series used as historical context.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables aligned with ``y``.
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon passed during fit.

        Returns
        -------
        self : TabPFNTSForecaster
            Reference to the fitted forecaster.
        """
        _check_soft_dependencies("tabpfn_time_series")
        cached_model = _CachedTabPFNTS(
            (self.tabpfn_mode, self.max_context_length, self.tabpfn_output_selection),
            self.tabpfn_mode,
            self.max_context_length,
            self.tabpfn_output_selection,
        )
        self.model_pipeline_ = cached_model.load_pipeline()

    def _run_pipeline(self, fh, X=None, alpha=None):
        """Run the TabPFN-TS pipeline and return raw forecast output.

        This helper constructs the TabPFN-TS input format from the stored
        training target series and optional exogenous variables, executes
        the underlying TabPFN-TS pipeline, and post-processes the result
        to align predictions with the requested forecasting horizon.

        The returned dataframe contains the point forecast column
        ("target") as well as quantile forecasts when requested or when
        produced by the underlying model by default.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon specifying the time points to predict.
        X : pd.DataFrame, optional (default=None)
            Future values of exogenous variables aligned with ``fh``.
        alpha : list of float, optional (default=None)
            Quantile levels to request from the TabPFN-TS pipeline.
            If ``None``, the pipeline uses its default quantile configuration.

        Returns
        -------
        pred_df : pd.DataFrame
            DataFrame containing the raw predictions produced by the
            TabPFN-TS pipeline. The index corresponds to the forecasting
            horizon, and the columns include the point forecast ("target")
            and any quantile forecasts returned by the model.
        """
        fh_rel = fh.to_relative(self.cutoff)
        prediction_length = int(fh_rel.max())

        context_df = self._y.copy()
        context_df = context_df.rename(columns={context_df.columns[0]: "target"})

        context_df = context_df.reset_index()
        context_df.rename(columns={context_df.columns[0]: "timestamp"}, inplace=True)

        if isinstance(context_df["timestamp"].dtype, pd.PeriodDtype):
            context_df["timestamp"] = context_df["timestamp"].dt.to_timestamp()

        predict_kwargs = {}
        if alpha is not None:
            predict_kwargs["quantiles"] = alpha

        if X is None:
            pred_df = self.model_pipeline_.predict_df(
                context_df=context_df,
                prediction_length=prediction_length,
                **predict_kwargs,
            )
        else:
            future_df = X.copy().reset_index()
            future_df.rename(columns={future_df.columns[0]: "timestamp"}, inplace=True)

            if isinstance(future_df["timestamp"].dtype, pd.PeriodDtype):
                future_df["timestamp"] = future_df["timestamp"].dt.to_timestamp()

            pred_df = self.model_pipeline_.predict_df(
                context_df=context_df, future_df=future_df, **predict_kwargs
            )

        fh_idx = fh_rel.to_numpy() - 1
        if X is None:
            pred_df = pred_df.iloc[fh_idx]
        else:
            pred_df = pred_df.iloc[list(range(len(fh_rel)))]

        fh_abs = fh.to_absolute(self.cutoff)
        pred_df.index = fh_abs.to_pandas()

        return pred_df

    def _predict_quantiles(self, alpha=None, **kwargs):
        """Compute quantile forecasts for the requested forecasting horizon.

        This method calls the TabPFN-TS pipeline via ``_run_pipeline`` and
        extracts the quantile forecast columns. The output is reformatted
        to comply with the ``sktime`` probabilistic forecasting interface,
        where columns follow a hierarchical MultiIndex structure with
        variable names at the first level and quantile levels at the second.

        Parameters
        ----------
        alpha : list of float, optional (default=None)
            Quantile levels at which forecasts are computed. If ``None``,
            the default quantiles returned by the TabPFN-TS pipeline are used.
        **kwargs : dict
            Additional keyword arguments passed internally by ``sktime``.
            Expected keys include:

            * ``fh`` : ForecastingHorizon
                Forecasting horizon specifying the time points to predict.
            * ``X`` : pd.DataFrame, optional
                Future exogenous variables aligned with ``fh``.

        Returns
        -------
        quantiles : pd.DataFrame
            Quantile forecasts with a MultiIndex column structure where:

            * Level 0 contains the target variable name.
            * Level 1 contains the quantile levels.

            The row index corresponds to the forecasting horizon.
        """
        fh = kwargs["fh"]
        X = kwargs.get("X", None)

        pred_df = self._run_pipeline(fh, X, alpha)
        y_name = self._y.columns[0]

        quantile_cols = [c for c in pred_df.columns if c != "target"]
        quantiles = [float(q) for q in quantile_cols]

        pred_df = pred_df[quantile_cols]
        pred_df.columns = pd.MultiIndex.from_product([[y_name], quantiles])

        return pred_df

    def _predict(self, fh, X=None):
        """Forecast time series for a given forecasting horizon.

        Uses the stored historical target series (and optional exogenous
        variables) to construct the TabPFN-TS context input and generate
        forecasts for the requested horizon.

        Parameters
        ----------
        fh : ForecastingHorizon
            Forecasting horizon specifying future time points to predict.
        X : pd.DataFrame, optional (default=None)
            Future values of exogenous variables aligned with ``fh``.

        Returns
        -------
        y_pred : pd.DataFrame
            Point forecasts indexed by the forecasting horizon.
        """
        y_name = self._y.columns[0]

        pred_df = self._run_pipeline(fh, X)
        point_pred = pred_df["target"].to_frame(name=y_name)
        return point_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return parameter settings for estimator testing.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the test parameter set.

        Returns
        -------
        params : list of dict
            Parameter settings used to create test instances of the estimator.
        """
        return [
            {"tabpfn_mode": "LOCAL"},
            {"tabpfn_mode": "CLIENT"},
            {
                "tabpfn_mode": "LOCAL",
                "max_context_length": 512,
                "tabpfn_output_selection": "median",
            },
        ]


@_multiton
class _CachedTabPFNTS:
    """Cached TabPFN-TS model, to ensure only one instance exists in memory.

    This is a zero shot model and immutable, hence there will not be
    any side effects of sharing the same instance across multiple uses.
    Plus, there are very minimal config parameters, which makes this simple.
    """

    def __init__(self, key, tabpfn_mode, max_context_length, tabpfn_output_selection):
        self.key = key
        self.tabpfn_mode = tabpfn_mode
        self.max_context_length = max_context_length
        self.tabpfn_output_selection = tabpfn_output_selection
        self.model_pipeline = None

    def load_pipeline(self):
        if self.model_pipeline is not None:
            return self.model_pipeline

        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        self.model_pipeline = TabPFNTSPipeline(
            tabpfn_mode=TabPFNMode[self.tabpfn_mode],
            max_context_length=self.max_context_length,
            tabpfn_output_selection=self.tabpfn_output_selection,
        )

        return self.model_pipeline
