import pandas as pd
from sktime.forecasting.base import BaseForecaster
from gluonts.dataset.common import ListDataset
from gluonts.mx.trainer import Trainer

from sktime.datatypes._adapter.gluonts import (
    convert_gluonts_result_to_multiindex,
    convert_from_multiindex_to_listdataset,
)


class GluonTSBaseAdapter(BaseForecaster):
    """
    Common adapter base class for GluonTS models.
    """

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
        self.model_kwargs = kwargs

        self.model = self._init_model()
        super().__init__()

    def _init_model(self):
        """
        Initialize the specific GluonTS model with a generic trainer.
        """
        trainer = Trainer(epochs=self.epochs)
        return self.model_class(
            freq=self.freq,
            prediction_length=self.prediction_length,
            trainer=trainer,
            **self.model_kwargs,
        )

    def _convert_to_gluonts_format(self, y):
        """
        Convert sktime data to GluonTS ListDataset format.
        """
        return convert_from_multiindex_to_listdataset(y)

    def _convert_to_sktime_format(self, gluonts_result):
        """
        Convert GluonTS forecast result to sktime format.
        """
        return convert_gluonts_result_to_multiindex(gluonts_result)

    def _fit(self, y, X=None, fh=None):
        """
        Fit the model to the training data.
        """
        gluonts_data = self._convert_to_gluonts_format(y)
        self.model.train(gluonts_data)
        return self

    def _predict(self, fh, X=None):
        """
        Generate predictions.
        """
        gluonts_data = self._convert_to_gluonts_format(self._y)
        forecast_it, ts_it = self.model.make_evaluation_predictions(
            gluonts_data, num_samples=100
        )
        forecast_entry = next(forecast_it)
        return self._convert_to_sktime_format(forecast_entry)

    def update(self, y_new, update_params=True):
        """
        Update or retrain the model with new data.
        """
        gluonts_data = self._convert_to_gluonts_format(y_new)
        self.model.train(gluonts_data)
        return self


class DeepARAdapter(GluonTSBaseAdapter):
    """
    Adapter for the GluonTS DeepAR model.
    """

    def __init__(self, **kwargs):
        from gluonts.model.deepar import DeepAREstimator

        super().__init__(model_class=DeepAREstimator, **kwargs)


class NBEATSAdapter(GluonTSBaseAdapter):
    """
    Adapter for the GluonTS N-BEATS model.
    """

    def __init__(self, **kwargs):
        from gluonts.model.n_beats import NBEATSEstimator

        super().__init__(model_class=NBEATSEstimator, **kwargs)