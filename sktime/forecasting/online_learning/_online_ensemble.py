# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.compose._ensemble import EnsembleForecaster
from sktime.forecasting.model_selection import SlidingWindowSplitter
from sktime.utils.validation.forecasting import check_cv
from sktime.utils.validation.forecasting import check_y


class OnlineEnsembleForecaster(EnsembleForecaster):
    """Online Updating Ensemble of forecasters

    Parameters
    ----------
    ensemble_algorithm : ensemble algorithm
    forecasters : list of (str, estimator) tuples
    n_jobs : int or None, optional (default=None)
        The number of jobs to run in parallel for fit. None means 1 unless
        in a joblib.parallel_backend context.
        -1 means using all processors.
    """

    _required_parameters = ["forecasters"]
    _tags = {
        "univariate-only": True,
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
    }

    def __init__(self, forecasters, ensemble_algorithm=None, n_jobs=None):

        self.n_jobs = n_jobs
        self.ensemble_algorithm = ensemble_algorithm

        super(EnsembleForecaster, self).__init__(forecasters=forecasters, n_jobs=n_jobs)

    def _fit(self, y, X=None, fh=None):
        """Fit to training data.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        fh : int, list or np.array, optional (default=None)
            The forecasters horizon with the steps ahead to to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        Returns
        -------
        self : returns an instance of self.
        """

        names, forecasters = self._check_forecasters()
        self.weights = np.ones(len(forecasters)) / len(forecasters)
        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _fit_ensemble(self, y, X=None):
        """Fits the ensemble by allowing forecasters to predict and
           compares to the actual parameters.

        Parameters
        ----------
        y : pd.Series
            Target time series to which to fit the forecaster.
        X : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        """
        fh = np.arange(len(y)) + 1
        estimator_predictions = np.column_stack(self._predict_forecasters(fh, X))
        y = np.array(y)

        self.ensemble_algorithm.update(estimator_predictions.T, y)

    def _update(self, y, X=None, update_params=False):
        """Update fitted paramters and performs a new ensemble fit.

        Parameters
        ----------
        y : pd.Series
        X : pd.DataFrame
        update_params : bool, optional (default=False)

        Returns
        -------
        self : an instance of self
        """

        if len(y) >= 1 and self.ensemble_algorithm is not None:
            self._fit_ensemble(y, X)

        for forecaster in self.forecasters_:
            forecaster.update(y, X, update_params=update_params)

        return self

    def update_predict(
        self,
        y_test,
        cv=None,
        X_test=None,
        update_params=False,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Make and update predictions iteratively over the test set.

        Parameters
        ----------
        y_test : pd.Series
        cv : temporal cross-validation generator, optional (default=None)
        X_test : pd.DataFrame, optional (default=None)
        update_params : bool, optional (default=False)
        return_pred_int : bool, optional (default=False)
        alpha : int or list of ints, optional (default=None)

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        y_pred_int : pd.DataFrame
            Prediction intervals
        """
        if return_pred_int:
            raise NotImplementedError()
        y_test = check_y(y_test)
        if cv is not None:
            cv = check_cv(cv)
        else:
            cv = SlidingWindowSplitter(start_with_window=True, window_length=1, fh=1)

        return self._predict_moving_cutoff(
            y_test,
            X=X_test,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
            cv=cv,
        )

    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()
        if self.ensemble_algorithm is not None:
            self.weights = self.ensemble_algorithm.weights
        return (pd.concat(self._predict_forecasters(fh, X), axis=1) * self.weights).sum(
            axis=1
        )
