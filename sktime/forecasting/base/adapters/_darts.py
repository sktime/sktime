# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for Darts models."""
import abc
from typing import List, Optional

import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

__author__ = ["yarnabrina", "fnhirwa"]


class _DartsAdapter(BaseForecaster):
    """Base adapter class for Darts models.

    Parameters
    ----------
    past_covariates : Optional[List[str]], optional
        column names in ``X`` which are known only for historical data, by default None
    num_samples : Optional[int], optional
        Number of times a prediction is sampled from a probabilistic model, by default
        1000

    Notes
    -----
    If unspecified, all columns will be assumed to be known during prediction duration.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["yarnabrina", "fnhirwa"],
        "maintainers": ["yarnabrina", "fnhirwa"],
        "python_version": ">=3.9",
        "python_dependencies": ["u8darts"],
        "python_dependencies_alias": {"u8darts": "darts"},
        # estimator type
        # --------------
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "requires-fh-in-fit": False,
        "enforce_index_type": pd.DatetimeIndex,
        "handles-missing-data": False,
        "capability:insample": False,
    }

    def __init__(
        self: "_DartsAdapter",
        past_covariates: Optional[List[str]] = None,
        num_samples: Optional[int] = 1000,
    ) -> None:
        if not isinstance(past_covariates, list) and past_covariates is not None:
            raise TypeError(
                f"Expected past_covariates to be a list, found {type(past_covariates)}."
            )
        self.past_covariates = [] if past_covariates is None else past_covariates
        if not isinstance(num_samples, int):
            raise TypeError(
                f"Expected num_samples to be an integer, found {type(num_samples)}."
            )
        self.num_samples = num_samples

        super().__init__()

        # initialize internal variables to avoid AttributeError
        self._forecaster = None

    @staticmethod
    def convert_dataframe_to_timeseries(dataset: pd.DataFrame):
        """Convert dataset for compatibility with ``darts``.

        Parameters
        ----------
        dataset : pandas.DataFrame
            source dataset to convert from

        Returns
        -------
        darts.TimeSeries
            converted target dataset
        """
        import darts

        dataset.index = dataset.index.astype("datetime64[ns]")  # ensure datetime index
        return darts.TimeSeries.from_dataframe(dataset)

    def convert_exogenous_dataset(
        self: "_DartsAdapter", dataset: Optional[pd.DataFrame]
    ):
        """Make exogenous features to ``darts`` compatible, if available.

        Parameters
        ----------
        dataset : Optional[pandas.DataFrame]
            available data on exogenous features

        Returns
        -------
        Tuple[darts.TimeSeries, darts.TimeSeries]
            converted data on future known and future unknown exogenous features
        """
        if dataset is None and self.past_covariates:
            raise ValueError(
                f"Expected following exogenous features: {self.past_covariates}."
            )

        if dataset is None:
            future_known_dataset = None
            future_unknown_dataset = None
        elif self.past_covariates:
            future_unknown_dataset = self.convert_dataframe_to_timeseries(
                dataset[self.past_covariates]
            )
            future_known_dataset = self.convert_dataframe_to_timeseries(
                dataset.drop(columns=self.past_covariates)
            )
        else:
            future_unknown_dataset = None
            future_known_dataset = self.convert_dataframe_to_timeseries(dataset)

        return future_known_dataset, future_unknown_dataset

    @abc.abstractmethod
    def _create_forecaster(self: "_DartsAdapter"):
        """Create Darts model."""

    def _fit(
        self: "_DartsAdapter",
        y: pd.DataFrame,
        X: Optional[pd.DataFrame],
        fh: Optional[ForecastingHorizon],
    ):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here.
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        del fh  # avoid being detected as unused by ``vulture`` like tools

        endogenous_actuals = self.convert_dataframe_to_timeseries(y)
        unknown_exogenous, known_exogenous = self.convert_exogenous_dataset(X)

        self._forecaster = self._create_forecaster()

        self._forecaster.fit(
            endogenous_actuals,
            past_covariates=unknown_exogenous,
            future_covariates=known_exogenous,
        )

        return self

    def _predict(
        self: "_DartsAdapter",
        fh: Optional[ForecastingHorizon],
        X: Optional[pd.DataFrame],
    ):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        unknown_exogenous, known_exogenous = self.convert_exogenous_dataset(X)

        maximum_forecast_horizon = fh.to_relative(self.cutoff)[-1]

        endogenous_point_predictions = self._forecaster.predict(
            maximum_forecast_horizon,
            past_covariates=unknown_exogenous,
            future_covariates=known_exogenous,
            num_samples=1,
        ).pd_dataframe()

        return endogenous_point_predictions

    # todo 0.22.0 - switch legacy_interface default to False
    # todo 0.23.0 - remove legacy_interface arg
    def _predict_quantiles(
        self,
        fh: Optional[ForecastingHorizon],
        X: Optional[pd.DataFrame],
        alpha: List[float],
        legacy_interface: Optional[bool] = True,
    ):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
        alpha : list of float, optional (default=[0.5])
            A list of probabilities at which quantile forecasts are computed.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        unknown_exogenous, known_exogenous = self.convert_exogenous_dataset(X)

        maximum_forecast_horizon = fh.to_relative(self.cutoff)[-1]

        endogenous_quantile_predictions = self._forecaster.predict(
            maximum_forecast_horizon,
            past_covariates=unknown_exogenous,
            future_covariates=known_exogenous,
            num_samples=self.num_samples,
        ).quantiles_df(quantiles=alpha)

        return endogenous_quantile_predictions


__all__ = ["_DartsAdapter"]
