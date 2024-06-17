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
        "capability:predint:insample": True,
    }

    def __init__(
        self: "_DartsAdapter",
        past_covariates: Optional[List[str]] = None,
        num_samples: Optional[int] = 1000,
    ) -> None:
        if past_covariates is not None and not isinstance(past_covariates, list):
            raise TypeError(
                f"Expected past_covariates to be a list, found {type(past_covariates)}."
            )
        self.past_covariates = past_covariates
        if num_samples is not None and not isinstance(num_samples, int):
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

        dataset_copy = _handle_input_index(dataset)
        return darts.TimeSeries.from_dataframe(dataset_copy)

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

        elif self.past_covariates is not None and self._lags_past_covariates:
            future_unknown_dataset = self.convert_dataframe_to_timeseries(
                dataset[self.past_covariates]
            )
            future_known_dataset = self.convert_dataframe_to_timeseries(
                dataset.drop(columns=self.past_covariates)
            )
        elif self._lags_future_covariates:
            future_unknown_dataset = None
            future_known_dataset = self.convert_dataframe_to_timeseries(dataset)
        else:
            future_known_dataset = None
            future_unknown_dataset = None

        return future_known_dataset, future_unknown_dataset

    @abc.abstractmethod
    def _create_forecaster(self: "_DartsAdapter"):
        """Create Darts model."""

    @property
    @abc.abstractmethod
    def _lags_past_covariates(self):
        """Get the lags_past_covariates value."""
        pass

    @property
    @abc.abstractmethod
    def _lags_future_covariates(self):
        """Get the lags_future_covariates value."""
        pass

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
        absolute_fh = fh.to_absolute(self.cutoff)
        maximum_forecast_horizon = fh.to_relative(self.cutoff)[-1]
        endogenous_point_predictions = self._forecaster.predict(
            maximum_forecast_horizon,
            past_covariates=unknown_exogenous,
            future_covariates=known_exogenous,
            num_samples=1,
        )
        original_index = fh.get_expected_pred_idx(self.cutoff)
        abs_idx = absolute_fh.to_pandas().astype(original_index.dtype)
        if self.get_class_tag("y_inner_mtype") == "pd.Series":
            # ToDo: handle the pd.Series
            pass
        else:
            endogenous_point_predictions = endogenous_point_predictions.pd_dataframe()

            if pd.api.types.is_integer_dtype(original_index):
                if X is not None:
                    if isinstance(X.index, pd.core.indexes.numeric.Int64Index):
                        endogenous_point_predictions.index = (
                            pd.core.indexes.numeric.Int64Index(
                                endogenous_point_predictions.index
                            )
                        )

            if isinstance(original_index, pd.PeriodIndex):
                endogenous_point_predictions.index = (
                    endogenous_point_predictions.index.to_period(original_index.freqstr)
                )
            if isinstance(original_index, pd.RangeIndex):
                endogenous_point_predictions.index = pd.RangeIndex(
                    start=0, stop=len(endogenous_point_predictions)
                )
            if isinstance(original_index, pd.DatetimeIndex):
                endogenous_point_predictions.index = pd.date_range(
                    start=original_index[0],
                    periods=len(endogenous_point_predictions),
                    freq=original_index.freq,
                )

            if (
                len(endogenous_point_predictions.columns) > 1
                and self._y.columns.dtype != "object"
            ):
                endogenous_point_predictions.columns = pd.RangeIndex(
                    start=0, stop=len(endogenous_point_predictions.columns), step=1
                )
            else:
                endogenous_point_predictions.columns = [
                    "c" + str(i) for i in range(endogenous_point_predictions.shape[1])
                ]
        return endogenous_point_predictions.loc[abs_idx]

    def _predict_quantiles(
        self,
        fh: Optional[ForecastingHorizon],
        X: Optional[pd.DataFrame],
        alpha: List[float],
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
        absolute_fh = fh.to_absolute(self.cutoff)
        endogenous_quantile_predictions = self._forecaster.predict(
            maximum_forecast_horizon,
            past_covariates=unknown_exogenous,
            future_covariates=known_exogenous,
            num_samples=self.num_samples,
        ).quantiles_df(quantiles=alpha)
        variable_names = self._get_varnames()
        multi_index = pd.MultiIndex.from_product(
            [variable_names, alpha], names=["variable", "quantile"]
        )
        original_index = fh.get_expected_pred_idx(self.cutoff)
        endogenous_quantile_predictions.index = (
            endogenous_quantile_predictions.index.astype(original_index.dtype)
        )

        abs_idx = absolute_fh.to_pandas().astype(original_index.dtype)
        endogenous_quantile_predictions.columns = multi_index
        return endogenous_quantile_predictions.loc[abs_idx]


def _handle_input_index(dataset: pd.DataFrame) -> pd.DataFrame:
    """Convert input dataset index to the compatible type for ``darts``.

    Parameters
    ----------
    dataset: pandas.DataFrame
        dataset with index to be converted

    Returns
    -------
    pandas.DataFrame
        converted dataset
    """
    if isinstance(dataset.index, (pd.DatetimeIndex, pd.RangeIndex)):
        return dataset
    dataset_copy = dataset.copy(deep=True)

    if isinstance(dataset_copy.index, pd.PeriodIndex):
        dataset_copy.index = dataset_copy.index.to_timestamp()
        return dataset_copy

    if pd.api.types.is_integer_dtype(dataset_copy.index):
        if isinstance(dataset_copy.index, pd.core.indexes.numeric.Int64Index):
            dataset_copy.index = pd.RangeIndex(
                start=dataset_copy.index.min(),
                stop=dataset_copy.index.max() + 1,
                step=dataset_copy.index[1] - dataset_copy.index[0],
            )
            return dataset_copy
        dataset_copy.index = pd.RangeIndex(start=0, stop=len(dataset_copy))
        return dataset_copy


__all__ = ["_DartsAdapter"]
