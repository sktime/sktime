# !/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements framework for applying online ensembling algorithms to forecasters."""

__author__ = ["magittan", "mloning"]

import numpy as np
import pandas as pd

from sktime.forecasting.compose._ensemble import EnsembleForecaster


class OnlineEnsembleForecaster(EnsembleForecaster):
    """Online Updating Ensemble of forecasters.

    Parameters
    ----------
    ensemble_algorithm : ensemble algorithm

    forecasters : list of estimator, (str, estimator), or (str, estimator, count) tuples
        Estimators to apply to the input series.

        * (str, estimator) tuples: the string is a name for the estimator.
        * estimator without string will be assigned unique name based on class name
        * (str, estimator, count) tuples: the estimator will be replicated count times.

    backend : {"dask", "loky", "multiprocessing", "threading"}, by default None.
        Runs parallel evaluate if specified and ``strategy`` is set as "refit".

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "dask_lazy": same as "dask",
          but changes the return to (lazy) ``dask.dataframe.DataFrame``.

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".
    backend_params : dict, optional
        additional parameters passed to the backend as config.
        Directly passed to ``utils.parallel.parallelize``.
        Valid keys depend on the value of ``backend``:

        - "None": no additional parameters, ``backend_params`` is ignored
        - "loky", "multiprocessing" and "threading": default ``joblib`` backends
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          with the exception of ``backend`` which is directly controlled by ``backend``.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``.
          any valid keys for ``joblib.Parallel`` can be passed here, e.g., ``n_jobs``,
          ``backend`` must be passed as a key of ``backend_params`` in this case.
          If ``n_jobs`` is not passed, it will default to ``-1``, other parameters
          will default to ``joblib`` defaults.
        - "dask": any valid keys for ``dask.compute`` can be passed,
          e.g., ``scheduler``
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["magittan", "mloning"],
        "maintainers": ["magittan"],
        # estimator type
        # --------------
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "y_inner_mtype": ["pd.Series"],
        "scitype:y": "univariate",
    }

    def __init__(
        self, 
        forecasters, 
        ensemble_algorithm=None,
        n_jobs="deprecated", 
        backend=None, 
        backend_params=None
    ):
        self.n_jobs = n_jobs
        self.ensemble_algorithm = ensemble_algorithm
        self.backend = backend
        self.backend_params = backend_params

        super().__init__(
            forecasters=forecasters,
            n_jobs=n_jobs,
            backend=backend, 
            backend_params=backend_params
        )

    def _fit(self, y, X, fh):
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
        forecasters = [x[1] for x in self.forecasters_]
        self.weights = np.ones(len(forecasters)) / len(forecasters)
        self._fit_forecasters(forecasters, y, X, fh)
        return self

    def _fit_ensemble(self, y, X=None):
        """Fit the ensemble.

        This makes predictions with individual forecasters and compares the
        results to actual values. This is then used to update ensemble
        weights.

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
        """Update fitted parameters and performs a new ensemble fit.

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

        for forecaster in self._get_forecaster_list():
            forecaster.update(y, X, update_params=update_params)

        return self

    def _predict(self, fh=None, X=None):
        if self.ensemble_algorithm is not None:
            self.weights = self.ensemble_algorithm.weights
        y_pred = pd.concat(self._predict_forecasters(fh, X), axis=1) * self.weights
        y_pred = y_pred.sum(axis=1)
        y_pred.name = self._y.name
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.


        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.naive import NaiveForecaster

        FORECASTER = NaiveForecaster()
        params1 = {"forecasters": [("f1", FORECASTER), ("f2", FORECASTER)]}

        params2 = {
            "forecasters": [
                ("f1", FORECASTER),
                ("f2", FORECASTER),
                ("f3", FORECASTER),
                ("f4", FORECASTER),
            ]
        }

        # the error log said it required atleast 2 values in the return value of
        # get_test_params
        params = [params1, params2]

        return params
