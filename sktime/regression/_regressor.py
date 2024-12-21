# Base class for regressors

__author__ = "Withington, James Large"

from sklearn.base import RegressorMixin
from sktime.regression.base import BaseRegressor

from sktime.utils import check_and_clean_data
from sktime.utils import check_is_fitted
from sktime.utils import save_trained_model


class BaseDeepRegressor(BaseRegressor, RegressorMixin):
    def __init__(self, model_name=None, model_save_directory=None):
        self.model_save_directory = model_save_directory
        self.model = None
        self.model_name = model_name

    def build_model(self, input_shape, **kwargs):
        """
        Construct a compiled, un-trained, keras model that is ready for
        training

        Parameters
        ----------
        input_shape : tuple The shape of the data fed
        into the input layer

        Returns
        -------
        output : a compiled Keras Model
        """
        raise NotImplementedError("this is an abstract method")

    def predict(self, X, input_checks=True, **kwargs):
        """
        Find regression estimate for all cases in X.
        Parameters
        ----------
        X : a nested pd.Dataframe, or (if input_checks=False) array-like of
        shape = (n_instances, series_length, n_dimensions)
            The training input samples. If a 2D array-like is passed,
            n_dimensions is assumed to be 1.
        input_checks: boolean
            whether to check the X parameter
        Returns
        -------
        predictions : 1d numpy array
            array of predictions of each instance
        """
        check_is_fitted(self)

        X = check_and_clean_data(X, input_checks=input_checks)

        y_pred = self.model.predict(X, **kwargs)

        if y_pred.ndim == 1:
            y_pred.ravel()
        return y_pred

    def save_trained_model(self):
        save_trained_model(
            self.model, self.model_save_directory, self.model_name
        )