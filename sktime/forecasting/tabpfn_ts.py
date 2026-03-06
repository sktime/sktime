"""Implements TabPFN-TS forecaster."""

__author__ = ["liam-sbhoo", "Infonioknight"]

# Required_deps = [autogluon-timeseries, backoff, datasets, fev, gluonts, pandas]
# [python-dotenv, pyyaml, statsmodels, tabpfn, tabpfn-client, tabpfn-common-utils, tqdm]

__all__ = ["TabPFNTSForecaster"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster


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
    >>> f.fit(y)
    >>> y_pred = f.predict(fh)
    """

    _tags = {
        "authors": ["liam-sbhoo", "Infonioknight"],
        # liam-sbhoo is from Prior Labs
        "maintainers": ["Infonioknight"],
        "python_dependencies": ["tabpfn-time-series"],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "requires-fh-in-fit": False,
        "capability:exogenous": True,
        "capability:global_forecasting": False,
    }

    def __init__(self, tabpfn_mode="CLIENT"):
        self.tabpfn_mode = tabpfn_mode
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
        _check_soft_dependencies("tabpfn-time-series")
        from tabpfn_time_series import TabPFNMode, TabPFNTSPipeline

        self.model_pipeline = TabPFNTSPipeline(tabpfn_mode=TabPFNMode[self.tabpfn_mode])
        return self

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
        prediction_length = int(max(fh.to_relative(self.cutoff))) if fh else 1

        context_df = pd.DataFrame(self._y).copy()
        context_df.columns = ["target"]

        if self._X is not None:
            context_df = context_df.join(self._X)

        context_df = context_df.reset_index()
        context_df = context_df.rename(columns={context_df.columns[0]: "timestamp"})

        if X is None:
            pred_df = self.model_pipeline.predict_df(
                context_df=context_df,
                prediction_length=prediction_length,
            )
        else:
            future_df = X.copy().reset_index()
            future_df = future_df.rename(columns={future_df.columns[0]: "timestamp"})

            pred_df = self.model_pipeline.predict_df(
                context_df=context_df,
                future_df=future_df,
            )

        point_pred = pred_df["0.5"] if "0.5" in pred_df.columns else pred_df[0.5]

        fh_abs = fh.to_absolute(self.cutoff)
        point_pred.index = fh_abs.to_pandas()

        return point_pred.to_frame(name=self._y.columns[0])

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
        return [{"tabpfn_mode": "LOCAL"}]
