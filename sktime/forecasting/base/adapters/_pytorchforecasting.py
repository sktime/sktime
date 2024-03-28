# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for pytorch-forecasting models."""
import abc
import functools
import typing
from typing import Any, Dict, List

import pandas
from pytorch_forecasting.data import TimeSeriesDataSet
from pytorch_forecasting.metrics import MultiHorizonMetric
from torch import nn

from sktime.forecasting.base import ForecastingHorizon, GlobalBaseForecaster

__all__ = ["_PytorchForecastingAdapter"]
__author__ = ["XinyuWu"]


class _PytorchForecastingAdapter(GlobalBaseForecaster):
    """Base adapter class for pytorch-forecasting models."""

    _tags = {
        # packaging info
        # --------------
        "authors": ["XinyuWu"],
        "maintainers": ["XinyuWu"],
        "python_version": ">=3.8",
        "python_dependencies": ["pytorch-forecasting"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "handles-missing-data": False,
        "capability:insample": False,
    }

    def __init__(
        self: "_PytorchForecastingAdapter",
        loss: MultiHorizonMetric = None,
        logging_metrics: nn.ModuleList = None,
        allowed_encoder_known_variable_names: List[str] | None = None,
        trainer_params: Dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.loss = loss
        self.logging_metrics = logging_metrics
        self.allowed_encoder_known_variable_names = allowed_encoder_known_variable_names
        self.trainer_params = trainer_params
        self._kwargs = kwargs

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_class(self: "_PytorchForecastingAdapter"):
        """Import underlying pytorch-forecasting algorithm class."""

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_parameters(self: "_PytorchForecastingAdapter") -> dict:
        """Keyword parameters for the underlying pytorch-forecasting algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class

        """

    def _instantiate_model(self: "_PytorchForecastingAdapter", data: TimeSeriesDataSet):
        """Instantiate the model."""
        algorithm_instance = self.algorithm_class.from_dataset(
            data,
            self.allowed_encoder_known_variable_names,
            **self.algorithm_parameters,
            **{"loss": self.loss, "logging_metrics": self.logging_metrics},
            **self._kwargs,
        )
        import pytorch_lightning as pl

        traner_instance = pl.Trainer(**self.trainer_params)
        return algorithm_instance, traner_instance

    def _fit(
        self: "_PytorchForecastingAdapter",
        y: pandas.Series,
        X: typing.Optional[pandas.DataFrame],
        fh: ForecastingHorizon,
    ) -> "_PytorchForecastingAdapter":
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to have a single column/variable
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to have at least one column/variable
            Exogeneous time series to fit to.

        Returns
        -------
        self : _PytorchForecastingAdapter
            reference to self

        Raises
        ------
        ValueError
            When ``freq="auto"`` and cannot be interpreted from ``ForecastingHorizon``
        """
        # TODO convert X to pytorch-forcasting TimeSeriesDataSet Xdataset
        self._forecaster, self._trainer = self._instantiate_model(X)
        # TODO convert X to pytorch-forcasting TimeSeriesDataSet.to_dataloader()
        self._trainer.fit(
            self._forecaster,
            train_dataloaders=X.to_dataloader(train=True),
            val_dataloaders=X.to_dataloader(train=False),
        )
        return self

    def _predict(
        self: "_PytorchForecastingAdapter",
        fh: typing.Optional[ForecastingHorizon],
        X: typing.Optional[pandas.DataFrame],
        y: typing.Optional[pandas.Series],
    ) -> pandas.Series:
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
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        y : sktime time series object, optional (default=None)
            Historical values of the time series that should be predicted.

        Returns
        -------
        y_pred : sktime time series object
            guaranteed to have a single column/variable
            Point predictions
        """
        # TODO convert X, y to pytorch-forecasting dataloader Xy_dataloader
        best_model_path = self._trainer.checkpoint_callback.best_model_path
        best_model = self.algorithm_class.load_from_checkpoint(best_model_path)
        predictions = best_model.predict(X, return_y=True)
        # TODO convert predictions to pandas.Series final_predictions
        return predictions