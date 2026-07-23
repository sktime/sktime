"""
Adapters for integrating GluonTS models with sktime's forecasting interface.

This module includes base and specific adapters to use GluonTS models, like
DeepAR and N-BEATS, in sktime workflows.
"""

import pandas as pd
from typing import Optional, Dict, Any

from sktime.forecasting.base import BaseForecaster
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer

from sktime.datatypes._adapter.gluonts import (
    convert_gluonts_result_to_multiindex,
    convert_from_multiindex_to_listdataset,
)


class _GluonTSBaseAdapter(BaseForecaster):
    """
    Common adapter base class for GluonTS models.

    This class serves as a base for integrating GluonTS models with sktime's forecasting framework.
    It handles the conversion between sktime's data formats and GluonTS's data formats,
    as well as the initialization and training of GluonTS models.
    """

    def __init__(
        self,
        model_class: Any,
        freq: str = "D",
        prediction_length: int = 24,
        epochs: int = 10,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the GluonTS base adapter.

        Parameters
        ----------
        model_class : class
            GluonTS model class to instantiate.
        freq : str, optional (default="D")
            Frequency of the time series.
        prediction_length : int, optional (default=24)
            Number of time steps to predict.
        epochs : int, optional (default=10)
            Number of training epochs.
        trainer_kwargs : dict, optional (default=None)
            Additional keyword arguments for the GluonTS Trainer.
        """
        self.freq: str = freq
        self.prediction_length: int = prediction_length
        self.epochs: int = epochs
        self.model_class: Any = model_class
        self.trainer_kwargs: Dict[str, Any] = trainer_kwargs or {}

        self.model = self._init_model()
        self._is_fitted: bool = False
        super().__init__()

    def _init_model(self) -> Any:
        """
        Initialize the specific GluonTS model with a generic trainer.

        Returns
        -------
        model : Any
            An instance of the GluonTS model.
        """
        trainer = Trainer(epochs=self.epochs, **self.trainer_kwargs)
        model = self.model_class(
            freq=self.freq,
            prediction_length=self.prediction_length,
            trainer=trainer,
        )
        return model

    def _convert_to_gluonts_format(self, y: pd.Series) -> ListDataset:
        """
        Convert sktime data to GluonTS ListDataset format.

        Parameters
        ----------
        y : pd.Series
            The time series data to convert.

        Returns
        -------
        gluonts_dataset : ListDataset
            The converted GluonTS ListDataset.
        """
        return convert_from_multiindex_to_listdataset(y)

    def _convert_to_sktime_format(self, gluonts_result: Any) -> pd.DataFrame:
        """
        Convert GluonTS forecast result to sktime format.

        Parameters
        ----------
        gluonts_result : Any
            The forecast result from GluonTS.

        Returns
        -------
        sktime_forecast : pd.DataFrame
            The converted forecast in sktime's MultiIndex format.
        """
        return convert_gluonts_result_to_multiindex(gluonts_result)

    def _fit(self, y: pd.Series, X: Optional[Any] = None, fh: Optional[Any] = None) -> "_GluonTSBaseAdapter":
        """
        Fit the model to the training data.

        Parameters
        ----------
        y : pd.Series
            The training time series.
        X : Optional[Any], optional (default=None)
            Optional exogenous variables.
        fh : Optional[Any], optional (default=None)
            Forecasting horizon.

        Returns
        -------
        self : _GluonTSBaseAdapter
            The fitted adapter instance.
        """
        gluonts_data = self._convert_to_gluonts_format(y)
        self.model.train(gluonts_data)
        self._is_fitted = True
        return self

    def _predict(self, fh: Any, X: Optional[Any] = None) -> pd.DataFrame:
        """
        Generate predictions.

        Parameters
        ----------
        fh : Any
            Forecasting horizon.
        X : Optional[Any], optional (default=None)
            Optional exogenous variables.

        Returns
        -------
        y_pred : pd.DataFrame
            The predicted values in sktime's MultiIndex format.
        """
        gluonts_data = self._convert_to_gluonts_format(self._y)
        forecast_it, _ = self.model.make_evaluation_predictions(
            gluonts_data, num_samples=100
        )
        forecast_entry = next(forecast_it)
        return self._convert_to_sktime_format(forecast_entry)

    def update(self, y_new: pd.Series, update_params: bool = True) -> "_GluonTSBaseAdapter":
        """
        Update or retrain the model with new data.

        Parameters
        ----------
        y_new : pd.Series
            The new time series data for updating the model.
        update_params : bool, optional (default=True)
            Whether to update the model parameters.

        Returns
        -------
        self : _GluonTSBaseAdapter
            The updated adapter instance.
        """
        gluonts_data = self._convert_to_gluonts_format(y_new)
        self.model.train(gluonts_data)
        return self


class DeepARAdapter(_GluonTSBaseAdapter):
    """
    Adapter for the GluonTS DeepAR model.

    This adapter integrates the DeepAR estimator from GluonTS with sktime's forecasting framework.
    """

    def __init__(
        self,
        freq: str = "D",
        prediction_length: int = 24,
        epochs: int = 10,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        # Add any DeepAR-specific parameters here
    ):
        """
        Initialize the DeepAR adapter.

        Parameters
        ----------
        freq : str, optional (default="D")
            Frequency of the time series.
        prediction_length : int, optional (default=24)
            Number of time steps to predict.
        epochs : int, optional (default=10)
            Number of training epochs.
        trainer_kwargs : dict, optional (default=None)
            Additional keyword arguments for the GluonTS Trainer.
        """
        from gluonts.model.deepar import DeepAREstimator

        super().__init__(
            model_class=DeepAREstimator,
            freq=freq,
            prediction_length=prediction_length,
            epochs=epochs,
            trainer_kwargs=trainer_kwargs,
        )
        # Initialize DeepAR-specific attributes if necessary


class NBEATSAdapter(_GluonTSBaseAdapter):
    """
    Adapter for the GluonTS N-BEATS model.

    This adapter integrates the N-BEATS estimator from GluonTS with sktime's forecasting framework.
    """

    def __init__(
        self,
        freq: str = "D",
        prediction_length: int = 24,
        epochs: int = 10,
        trainer_kwargs: Optional[Dict[str, Any]] = None,
        # Add any N-BEATS-specific parameters here
    ):
        """
        Initialize the N-BEATS adapter.

        Parameters
        ----------
        freq : str, optional (default="D")
            Frequency of the time series.
        prediction_length : int, optional (default=24)
            Number of time steps to predict.
        epochs : int, optional (default=10)
            Number of training epochs.
        trainer_kwargs : dict, optional (default=None)
            Additional keyword arguments for the GluonTS Trainer.
        """
        from gluonts.model.n_beats import NBEATSEstimator

        super().__init__(
            model_class=NBEATSEstimator,
            freq=freq,
            prediction_length=prediction_length,
            epochs=epochs,
            trainer_kwargs=trainer_kwargs,
        )
        # Initialize N-BEATS-specific attributes if necessary