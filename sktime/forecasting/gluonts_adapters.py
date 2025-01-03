"""
Adapters for integrating GluonTS models with sktime's forecasting interface.

This module includes base and specific adapters to use GluonTS models, like
DeepAR and N-BEATS, in sktime workflows.
"""

from sktime.datatypes._adapter.gluonts_to_pd_multiindex import (
    convert_gluonts_result_to_multiindex,
)
from sktime.datatypes._adapter.pd_multiindex_to_list_dataset import (
    convert_from_multiindex_to_listdataset,
)

from sktime.forecasting.base._base import BaseForecaster


class GluonTSBaseAdapter(BaseForecaster):
    """Common adapter base class for GluonTS models."""

    def __init__(self, model_class, freq="D", prediction_length=24, epochs=10, **kwargs):
        """
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
        kwargs : dict
            Additional keyword arguments for the GluonTS model.
        """
        self.freq = freq
        self.prediction_length = prediction_length
        self.epochs = epochs
        self.model_class = model_class
        self.kwargs = kwargs
        self.model = self._init_model()
        super().__init__()

    def _init_model(self):
        """Initialize the specific GluonTS model with a PyTorch Lightning Trainer."""
        trainer_kwargs = {"max_epochs": self.epochs, "accelerator": "cpu"}  # Adjust for GPU if needed
        return self.model_class(
            freq=self.freq,
            prediction_length=self.prediction_length,
            trainer_kwargs=trainer_kwargs,
            **self.kwargs
        )

    def _convert_to_gluonts_format(self, y):
        """Convert sktime data to GluonTS ListDataset format."""
        return convert_from_multiindex_to_listdataset(y)

    def _convert_to_sktime_format(self, gluonts_result):
        """Convert GluonTS forecast result to sktime format."""
        return convert_gluonts_result_to_multiindex(gluonts_result)

    def _fit(self, y, X=None, fh=None):
        """Fit the model to the training data."""
        gluonts_data = self._convert_to_gluonts_format(y)
        self.model.train(gluonts_data)
        return self

    def _predict(self, fh, X=None):
        """Generate predictions."""
        gluonts_data = self._convert_to_gluonts_format(self._y)
        forecast_it, ts_it = self.model.make_evaluation_predictions(gluonts_data, num_samples=100)
        forecast_entry = next(forecast_it)
        return self._convert_to_sktime_format(forecast_entry)

    def update(self, y_new, update_params=True):
        """Update or retrain the model with new data."""
        gluonts_data = self._convert_to_gluonts_format(y_new)
        self.model.train(gluonts_data)
        return self


class DeepARAdapter(GluonTSBaseAdapter):
    """Adapter for the GluonTS DeepAR model."""

    def __init__(self, **kwargs):
        from gluonts.torch.model.deepar import DeepAREstimator
        super().__init__(model_class=DeepAREstimator, **kwargs)


class NBEATSAdapter(GluonTSBaseAdapter):
    """Adapter for the GluonTS N-BEATS model."""

    def __init__(self, **kwargs):
        from gluonts.mx.model.n_beats import NBEATSEstimator  # MXNet-based implementation
        super().__init__(model_class=NBEATSEstimator, **kwargs)
