from .base import BaseFoundationForecaster
import numpy as np
from sklearn.ensemble import RandomForestRegressor


class TabPFNTimeSeriesForecaster(BaseFoundationForecaster):
    """
    TabPFN-inspired time series forecaster using tabular transformation.
    """

    def __init__(self, window_length=5, device=None):
        self.window_length = window_length
        self.model = RandomForestRegressor()
        super().__init__(model_name="tabpfn-ts", device=device)

    def load_model(self):
        # No external model needed
        pass

    def preprocess(self, y):
        values = y.values.astype(float)

        X, y_out = [], []

        for i in range(len(values) - self.window_length):
            X.append(values[i:i+self.window_length])
            y_out.append(values[i+self.window_length])

        return np.array(X), np.array(y_out)

    def _fit(self, y, X=None, fh=None):
        self._y = y

        X_train, y_train = self.preprocess(y)
        self.model.fit(X_train, y_train)

        return self

    def _infer(self, inputs, fh):
        values = self._y.values.astype(float).tolist()

        preds = []

        for _ in fh:
            x_input = np.array(values[-self.window_length:]).reshape(1, -1)
            pred = self.model.predict(x_input)[0]

            preds.append(pred)
            values.append(pred)

        return preds

    def postprocess(self, preds):
        return preds