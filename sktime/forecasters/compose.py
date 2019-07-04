import pandas as pd
import warnings
import numpy as np
from sklearn.base import clone
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import FunctionTransformer

from sktime.forecasters.base import BaseForecaster


class EnsembleForecaster(BaseForecaster):
    """
    Ensemble of forecasters.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        List of (name, estimator) tuples.
    weights : array-like, shape = [n_estimators], optional (default=None)
        Sequence of weights (float or int) to weight the occurrences of predicted values before averaging.
        Uses uniform weights if None.
    check_input : bool, optional (default=True)
        - If True, input are checked.
        - If False, input are not checked and assumed correct. Use with caution.
    """

    # TODO: experimental, major functionality not implemented (input checks, params interface, exogenous variables)

    def __init__(self, estimators=None, weights=None, check_input=True):
        # TODO add input checks
        self.estimators = estimators
        self.weights = weights
        self.fitted_estimators_ = []
        super(EnsembleForecaster, self).__init__(check_input=check_input)

    def _fit(self, y, fh=None, X=None):
        """
        Internal fit.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to fit the forecaster.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that
            if an ``ARIMA`` is fit on exogenous features, it must also be provided
            exogenous features for making predictions.

        Returns
        -------
        self : returns an instance of self.
        """
        for _, estimator in self.estimators:
            # TODO implement set/get params interface
            # estimator.set_params(**{"check_input": False})
            fitted_estimator = estimator.fit(y, fh=fh, X=X)
            self.fitted_estimators_.append(fitted_estimator)
        return self

    def _predict(self, fh=None, X=None):
        """
        Internal predict using fitted estimator.

        Parameters
        ----------
        fh : array-like, optional (default=None)
            The forecasters horizon with the steps ahead to to predict. Default is one-step ahead forecast,
            i.e. np.array([1])
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables. If provided, these
            variables are used as additional features in the regression
            operation. This should not include a constant or trend. Note that if
            provided, the forecaster must also have been fitted on the exogenous
            features.

        Returns
        -------
        Predictions : pandas.Series, shape=(len(fh),)
            Returns series of predicted values.
        """
        # TODO pass X only to estimators where the predict method accepts X, currenlty X is ignored

        # Forecast all periods from start to end of pred horizon, but only return given time points in pred horizon
        fh_idx = fh - np.min(fh)

        # Iterate over estimators
        y_preds = np.zeros((len(self.fitted_estimators_), len(fh)))
        indexes = []
        for i, estimator in enumerate(self.fitted_estimators_):
            y_pred = estimator.predict(fh=fh)
            y_preds[i, :] = y_pred
            indexes.append(y_pred.index)

        # Check if all predicted horizons are identical
        if not all(index.equals(indexes[0]) for index in indexes):
            raise ValueError('Predicted horizons from estimators do not match')

        # Average predictions over estimators
        avg_preds = np.average(y_preds, axis=0, weights=self.weights)

        # Return average predictions with index
        index = indexes[0]
        name = y_preds[0].name if hasattr(y_preds[0], 'name') else None
        return pd.Series(avg_preds, index=index, name=name)

    def get_params(self, deep=True):
        # TODO fix get and set params interface following sklearn double underscore convention
        raise NotImplementedError()

    def set_params(self, **params):
        raise NotImplementedError()


class TransformedTargetForecaster(BaseForecaster):
    """Meta-estimator to forecast on a transformed target."""

    # TODO add check inverse method after fitting transformer

    def __init__(self, forecaster, transformer, check_input=True):
        self.forecaster = forecaster
        self.transformer = transformer
        super(TransformedTargetForecaster, self).__init__(check_input=check_input)

    def _transform(self, y):
        # transformers are designed to modify X which is 2-dimensional, we
        # need to modify y accordingly.
        y = pd.DataFrame(y) if isinstance(y, pd.Series) else y

        self.transformer_ = clone(self.transformer)
        yt = self.transformer_.fit_transform(y)

        # restore 1d target
        yt = yt.iloc[:, 0]
        return yt

    def _inverse_transform(self, y):
        # transformers are designed to modify X which is 2-dimensional, we
        # need to modify y accordingly.
        y = pd.DataFrame(y) if isinstance(y, pd.Series) else y
        yit = self.transformer_.inverse_transform(y)

        # restore 1d target
        yit = yit.iloc[:, 0]
        return yit

    def _fit(self, y, fh=None, X=None):
        """Internal fit"""
        # store the number of dimension of the target to predict an array of
        # similar shape at predict
        self._input_shape = y.ndim

        yt = self._transform(y)

        # fit forecaster using transformed target data
        self.forecaster_ = clone(self.forecaster)
        self.forecaster_.fit(yt, fh=fh, X=X)
        return self

    def _predict(self, fh, X=None):
        """Internal predict"""
        check_is_fitted(self, "forecaster_")
        y_pred = self.forecaster_.predict(fh=fh, X=X)

        # return to nested format
        y_pred = pd.Series([y_pred])

        # compute inverse transform
        y_pred_it = self._inverse_transform(y_pred)

        # return unnested format
        return y_pred_it.iloc[0]
