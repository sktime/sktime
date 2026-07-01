"""
Improving time series forecasting accuracy.

ReconcilerForecaster improves time series forecasts.
"""

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster


class ReconcilerForecaster(BaseForecaster):
    """
    Reconciles forecasts using ML.

    Combines forecasts with ML for better accuracy.
    """

    def __init__(self, base_forecaster, regressor, cv):
        """
        Initialize forecaster, regressor, and cross-validator.

        Sets up base forecaster, regressor, and cross-validation for forecasting tasks.
        """
        self.base_forecaster = base_forecaster
        self.regressor = regressor
        self.cv = cv
        self._converter_store_y = {}
        self._fh = None

    def fit(self, y, X=None, fh=None):
        """
        Fit the ReconcilerForecaster using training data and base.

        forecaster predictions.
        """
        # Ensure y is a pandas Series
        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        base_preds = []

        for train_idx, test_idx in self.cv.split(y):
            y_train = y.iloc[train_idx]
            fh_test = np.arange(1, len(test_idx) + 1)

            self.base_forecaster.fit(y_train)
            preds = self.base_forecaster.predict(fh_test)
            base_preds.append(preds)

        base_preds = np.concatenate(base_preds)

        # Fit the regressor using base predictions
        self.regressor.fit(base_preds.reshape(-1, 1), y)

    def predict(self, fh, X=None):
        """
        Adjust predictions using regressor.

        Generates and adjusts predictions with forecaster and regressor.
        """
        # Get base forecaster predictions
        base_preds = self.base_forecaster.predict(fh)
        # Use regressor to adjust predictions
        return self.regressor.predict(base_preds.reshape(-1, 1))
