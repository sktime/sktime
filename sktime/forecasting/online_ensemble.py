import numpy as np
import pandas as pd
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.compose._ensemble import EnsembleForecaster
from sktime.forecasting.model_selection import OnlineSplitter
from .ensemble_algorithms import EnsembleAlgorithms


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

    def __init__(self, forecasters, ensemble_algorithm=None, n_jobs=None):
        self.n_jobs = n_jobs
        if ensemble_algorithm is None:
            self.ensemble_algorithm = EnsembleAlgorithms(len(forecasters))
        else:
            self.ensemble_algorithm = ensemble_algorithm

#         if self.ensemble_algorithm.n != len(forecasters):
#             raise ValueError("Number of Experts in Ensemble Algorithm \
#                              doesn't equal number of Forecasters")

        super(EnsembleForecaster, self).__init__(forecasters=forecasters,
                                                 n_jobs=n_jobs)

    def _fit_ensemble(self, y_val, X_val=None):
        """Fits the ensemble by allowing forecasters to predict and
           compares to the actual parameters.

        Parameters
        ----------
        y_val : pd.Series
            Target time series to which to fit the forecaster.
        X_val : pd.DataFrame, optional (default=None)
            Exogenous variables are ignored
        """
        fh = np.arange(len(y_val)) + 1
        expert_predictions = np.column_stack(self._predict_forecasters(
                                             fh=fh, X=X_val))
        actual_values = np.array(y_val)

        self.ensemble_algorithm._update(expert_predictions.T, actual_values)

    def update(self, y_new, X_new=None, update_params=False):
        """Update fitted paramters and performs a new ensemble fit.

        Parameters
        ----------
        y_new : pd.Series
        X_new : pd.DataFrame
        update_params : bool, optional (default=False)

        Returns
        -------
        self : an instance of self
        """
        self._fit_ensemble(y_new, X_val=X_new)

        self.check_is_fitted()
        self._set_oh(y_new)
        for forecaster in self.forecasters_:
            forecaster.update(y_new, X_new=X_new, update_params=update_params)

        return self

    def update_predict(self, y_test, X_test=None, update_params=False,
                       return_pred_int=False,
                       alpha=DEFAULT_ALPHA):
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

        return self._predict_moving_cutoff(y_test, X=X_test,
                                           update_params=update_params,
                                           return_pred_int=return_pred_int,
                                           alpha=alpha)

    def _predict_moving_cutoff(self, y, cv=OnlineSplitter(), X=None,
                               update_params=False,
                               return_pred_int=False,
                               alpha=DEFAULT_ALPHA):
        """Make single-step or multi-step moving cutoff predictions

        Parameters
        ----------
        y : pd.Series
        cv : temporal cross-validation generator, set to OnlineSplitter()
        X : pd.DataFrame
        update_params : bool
        return_pred_int : bool
        alpha : float or array-like

        Returns
        -------
        y_pred = pd.Series
        """
        if return_pred_int:
            raise NotImplementedError()
        fh = np.array([1])
        y_preds = []
        cutoffs = []
        with self._detached_cutoff():
            self._set_cutoff(y.index[0] - 1)
            for new_window, _ in cv.split(y):
                y_new = y.iloc[new_window]

                # we cannot use update_predict_single here, as this would
                # re-set the forecasting horizon, instead we use
                # the internal _update_predict_single method
                y_pred = self._update_predict_single(
                    y_new, fh, X=X,
                    update_params=update_params,
                    return_pred_int=return_pred_int,
                    alpha=alpha)
                y_preds.append(y_pred)
                cutoffs.append(self.cutoff)
        return _format_moving_cutoff_predictions(y_preds, cutoffs)

    def _predict(self, fh, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        if return_pred_int:
            raise NotImplementedError()

        return (pd.concat(
            self._predict_forecasters(fh=fh, X=X), axis=1)
                * self.ensemble_algorithm.weights).sum(axis=1)


def _format_moving_cutoff_predictions(y_preds, cutoffs):
    """Format moving-cutoff predictions"""
    if not isinstance(y_preds, list):
        raise ValueError(
            f"`y_preds` must be a list, but found: {type(y_preds)}")

    if len(y_preds[0]) == 1:
        # return series for single step ahead predictions
        return pd.concat(y_preds)

    else:
        # return data frame when we predict multiple steps ahead
        y_pred = pd.DataFrame(y_preds).T
        y_pred.columns = cutoffs
        if y_pred.shape[1] == 1:
            return y_pred.iloc[:, 0]
        return y_pred
