import torch
from chronos import ChronosPipeline
from .base import BaseFoundationForecaster
import pandas as pd


class ChronosForecaster(BaseFoundationForecaster):
    """
    Chronos-based time series forecaster using the official ChronosPipeline.
    """

    def __init__(self, model_name="amazon/chronos-t5-small", mode="zero-shot", device=None):
        super().__init__(model_name, device)
        self.mode = mode

    def load_model(self):
        self.pipeline = ChronosPipeline.from_pretrained(
            self.model_name,
            device_map=self.device or "cpu",
            dtype=torch.float32,
        )

    def _fit(self, y, X=None, fh=None):
        self.load_model()
        self._y = y
        if self.mode == "fine-tune":
            self.fine_tune(y)
        return self

    def _predict(self, fh, X=None):
        self.fh_len = len(fh)
        context = torch.tensor(self._y.values.astype(float), dtype=torch.float32)
        forecast_samples = self.pipeline.predict(
            context.unsqueeze(0),
            prediction_length=self.fh_len,
            num_samples=20,
        )
        preds = forecast_samples[0].median(dim=0).values.numpy().tolist()
        preds = [max(min(v, 1e4), -1e4) for v in preds]
        fh_index = self.fh.to_absolute(self.cutoff).to_pandas()
        return pd.Series(preds, index=fh_index)
