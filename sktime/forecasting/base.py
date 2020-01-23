__all__ = ["ForecasterMixin"]
__author__ = ["Markus Löning"]

import pandas as pd

from sktime.utils.validation.forecasting import validate_cv
from sktime.utils.validation.forecasting import validate_time_index
from sktime.utils.validation.forecasting import validate_y


class ForecasterMixin:
    """
    Base class for forecasters.
    """
    _estimator_type = "forecaster"

    def __init__(self):
        self._time_index = None  # forecasters need to keep track of time index of target series
        self.is_fitted = False
        self.fh = None

    def _update_time_index(self, time_index):
        if self._time_index is None:
            raise ValueError("Cannot update time index because no previous time index found")
        time_index = validate_time_index(time_index)

        new_index = self._time_index.append(time_index)
        if not new_index.is_monotonic:
            raise ValueError("Updated time index is no longer monotonically increasing. Data passed "
                             "to `update` must contain more recent data than data passed to `fit`.")
        return new_index

    def fit(self, y, fh=None, X=None):
        raise NotImplementedError

    def _check_is_fitted(self):
        raise NotImplementedError

    def predict(self, fh=None, X=None, return_conf_int=False, alpha=0.05):
        raise NotImplementedError

    def update(self, y_new, X_new=None, update_params=False):
        raise NotImplementedError

    def predict_update(self, y_test, cv, X_test=None, update_params=False, return_conf_int=False, alpha=0.05):
        if return_conf_int:
            raise NotImplementedError

        self._check_is_fitted()
        y_test = validate_y(y_test)
        cv = validate_cv(cv)

        fh = cv.fh
        step_length = cv.step_length
        window_length = cv.window_length

        # create new index to make first prediction at end of training set
        index = self._time_index[-window_length:].append(y_test.index).values

        # allocate lists for prediction results
        predictions = []
        prediction_timepoints = []  # time points at which we predict

        # iterative predict and update
        for i, (in_window, out_window) in enumerate(cv.split(index)):
            # first prediction from training set without updates
            if i == 0:
                y_pred = self.predict(fh)
                predictions.append(y_pred)
                prediction_timepoints.append(self._time_index[-1])
                continue

            new_window = in_window[-step_length:]
            y_new = y_test.loc[new_window]
            self.update(y_new)

            y_pred = self.predict(fh)
            predictions.append(y_pred)
            prediction_timepoints.append(self._time_index[-1])

        # concatenate predictions
        if len(fh) > 1:
            predictions = pd.DataFrame(predictions).T
            predictions.columns = prediction_timepoints
        else:
            predictions = pd.concat(predictions)
        return predictions

    def score(self, y_test, fh=None, X_test=None):
        raise NotImplementedError

    def score_update(self, y_test, cv=None, X_test=None, update_params=False):
        raise NotImplementedError
