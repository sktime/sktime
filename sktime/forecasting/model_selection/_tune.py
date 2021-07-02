#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["ForecastingGridSearchCV", "ForecastingRandomizedSearchCV"]

import pandas as pd
from joblib import Parallel
from joblib import delayed
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import _check_param_grid
from sklearn.utils.metaestimators import if_delegate_has_method

from sktime.exceptions import NotFittedError
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.forecasting.model_evaluation import evaluate
from sktime.utils.validation.forecasting import check_scoring


class BaseGridSearch(BaseForecaster):

    _tags = {
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "univariate-only": True,
    }

    def __init__(
        self,
        forecaster,
        cv,
        strategy="refit",
        n_jobs=None,
        pre_dispatch=None,
        refit=False,
        scoring=None,
        verbose=0,
    ):
        self.forecaster = forecaster
        self.cv = cv
        self.strategy = strategy
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        super(BaseGridSearch, self).__init__()

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def _update(self, y, X=None, update_params=False):
        """Call predict on the forecaster with the best found parameters."""
        self.check_is_fitted("update")
        self.best_forecaster_._update(y, X, update_params=update_params)
        return self

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def _update_predict(
        self,
        y,
        cv=None,
        X=None,
        update_params=False,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Call update_predict on the forecaster with the best found
        parameters.
        """
        self.check_is_fitted("update_predict")

        return self.best_forecaster_._update_predict(
            y,
            cv=cv,
            X=X,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def _update_predict_single(
        self,
        y,
        fh=None,
        X=None,
        update_params=False,
        return_pred_int=False,
        alpha=DEFAULT_ALPHA,
    ):
        """Call predict on the forecaster with the best found parameters."""
        self.check_is_fitted("update_predict_single")
        return self.best_forecaster_._update_predict_single(
            y,
            fh=fh,
            X=X,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def _predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Call predict on the forecaster with the best found parameters."""
        self.check_is_fitted("predict")
        return self.best_forecaster_._predict(
            fh, X, return_pred_int=return_pred_int, alpha=alpha
        )

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def _compute_pred_int(self, y_pred, alpha=DEFAULT_ALPHA):
        """Call compute_pred_int on the forecaster with the best found parameters."""
        self.check_is_fitted("compute_pred_int")
        return self.best_forecaster_._compute_pred_int(y_pred, alpha=alpha)

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def transform(self, y, X=None):
        """Call transform on the forecaster with the best found parameters."""
        self.check_is_fitted("transform")
        return self.best_forecaster_.transform(y, X)

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def get_fitted_params(self):
        """Get fitted parameters

        Returns
        -------
        fitted_params : dict
        """
        self.check_is_fitted("get_fitted_params")
        return self.best_forecaster_.get_fitted_params()

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def inverse_transform(self, y, X=None):
        """Call inverse_transform on the forecaster with the best found params.
        Only available if the underlying forecaster implements
        ``inverse_transform`` and ``refit=True``.
        Parameters
        ----------
        y : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying forecaster.
        """
        self.check_is_fitted("inverse_transform")
        return self.best_forecaster_.inverse_transform(y, X)

    def score(self, y, X=None, fh=None):
        """Returns the score on the given data, if the forecaster has been
        refit.
        This uses the score defined by ``scoring`` where provided, and the
        ``best_forecaster_.score`` method otherwise.

        Parameters
        ----------
        y : pandas.Series
            Target time series to which to compare the forecasts.
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.
        fh : ForecastingHorizon, int, np.ndarray, pd.Index, optional (default=None)
            Forecasting horizon

        Returns
        -------
        score : float
        """
        self.check_is_fitted("score")

        if self.scoring is None:
            return self.best_forecaster_.score(y, X=X, fh=fh)

        else:
            y_pred = self.best_forecaster_.predict(fh, X=X)
            return self.scoring(y, y_pred)

    def _run_search(self, evaluate_candidates):
        raise NotImplementedError("abstract method")

    def check_is_fitted(self, method_name=None):
        """Has `fit` been called?

        Parameters
        ----------
        method_name : str
            Name of the calling method.

        Raises
        ------
        NotFittedError
            If forecaster has not been fitted yet.
        """
        super(BaseGridSearch, self).check_is_fitted()

        # We additionally check if the tuned forecaster has been fitted.
        if method_name is not None:
            if not self.refit:
                raise NotFittedError(
                    "This %s instance was initialized "
                    "with refit=False. %s is "
                    "available only after refitting on the "
                    "best parameters. You can refit an forecaster "
                    "manually using the ``best_params_`` "
                    "attribute" % (type(self).__name__, method_name)
                )
            else:
                self.best_forecaster_.check_is_fitted()

    def _fit(self, y, X=None, fh=None, **fit_params):
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
        cv = check_cv(self.cv)
        scoring = check_scoring(self.scoring)
        scoring_name = f"test_{scoring.name}"

        parallel = Parallel(n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch)

        def _fit_and_score(params):
            # Clone forecaster.
            forecaster = clone(self.forecaster)

            # Set parameters.
            forecaster.set_params(**params)

            # Evaluate.
            out = evaluate(
                forecaster,
                cv,
                y,
                X,
                strategy=self.strategy,
                scoring=scoring,
                fit_params=fit_params,
            )

            # Filter columns.
            out = out.filter(items=[scoring_name, "fit_time", "pred_time"], axis=1)

            # Aggregate results.
            out = out.mean()
            out = out.add_prefix("mean_")

            # Add parameters to output table.
            out["params"] = params

            return out

        def evaluate_candidates(candidate_params):
            candidate_params = list(candidate_params)

            if self.verbose > 0:
                n_candidates = len(candidate_params)
                n_splits = cv.get_n_splits(y)
                print(  # noqa
                    "Fitting {0} folds for each of {1} candidates,"
                    " totalling {2} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits
                    )
                )

            out = parallel(
                delayed(_fit_and_score)(params) for params in candidate_params
            )

            if len(out) < 1:
                raise ValueError(
                    "No fits were performed. "
                    "Was the CV iterator empty? "
                    "Were there no candidates?"
                )

            return out

        # Run grid-search cross-validation.
        results = self._run_search(evaluate_candidates)

        results = pd.DataFrame(results)

        # Rank results, according to whether greater is better for the given scoring.
        results[f"rank_{scoring_name}"] = results.loc[:, f"mean_{scoring_name}"].rank(
            ascending=~scoring.greater_is_better
        )
        self.cv_results_ = results

        # Select best parameters.
        self.best_index_ = results.loc[:, f"rank_{scoring_name}"].argmin()
        self.best_score_ = results.loc[self.best_index_, f"mean_{scoring_name}"]
        self.best_params_ = results.loc[self.best_index_, "params"]
        self.best_forecaster_ = clone(self.forecaster).set_params(**self.best_params_)

        # Refit model with best parameters.
        if self.refit:
            self.best_forecaster_.fit(y, X, fh)

        return self


class ForecastingGridSearchCV(BaseGridSearch):
    """
    Performs grid-search cross-validation to find optimal model parameters.
    The forecaster is fit on the initial window and then temporal
    cross-validation is used to find the optimal parameter

    Grid-search cross-validation is performed based on a cross-validation
    iterator encoding the cross-validation scheme, the parameter grid to
    search over, and (optionally) the evaluation metric for comparing model
    performance. As in scikit-learn, tuning works through the common
    hyper-parameter interface which allows to repeatedly fit and evaluate
    the same forecaster with different hyper-parameters.

    Parameters
    ----------
    forecaster : estimator object
        The estimator should implement the sktime or scikit-learn estimator
        interface. Either the estimator must contain a "score" function,
        or a scoring function must be passed.
    cv : cross-validation generator or an iterable
        e.g. SlidingWindowSplitter()
    param_grid : dict or list of dictionaries
        Model tuning parameters of the forecaster to evaluate
    scoring: function, optional (default=None)
        Function to score models for evaluation of optimal parameters
    n_jobs: int, optional (default=None)
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    refit: bool, optional (default=True)
        Refit the forecaster with the best parameters on all the data
    verbose: int, optional (default=0)
    pre_dispatch: str, optional (default='2*n_jobs')
    error_score: numeric value or the str 'raise', optional (default=np.nan)
        The test score returned when a forecaster fails to be fitted.
    return_train_score: bool, optional (default=False)

    Attributes
    ----------
    best_index_ : int
    best_score_: float
        Score of the best model
    best_params_ : dict
        Best parameter values across the parameter grid
    best_forecaster_ : estimator
        Fitted estimator with the best parameters
    cv_results_ : dict
        Results from grid search cross validation
    n_splits_: int
        Number of splits in the data for cross validation}
    refit_time_ : float
        Time (seconds) to refit the best forecaster
    scorer_ : function
        Function used to score model

    Example
    ----------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.model_selection import (
    ...     ExpandingWindowSplitter,
    ...     ForecastingGridSearchCV,
    ...     ExpandingWindowSplitter)
    >>> from sktime.forecasting.naive import NaiveForecaster

    >>> y = load_airline()
    >>> fh = [1,2,3]
    >>> cv = ExpandingWindowSplitter(
    ...     start_with_window=True,
    ...     fh=fh)
    >>> forecaster = NaiveForecaster()
    >>> param_grid = {"strategy" : ["last", "mean", "drift"]}
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=forecaster,
    ...     param_grid=param_grid,
    ...     cv=cv)
    >>> gscv.fit(y)
    ForecastingGridSearchCV(...)
    >>> y_pred = gscv.predict(fh)
    """

    _required_parameters = ["forecaster", "cv", "param_grid"]

    def __init__(
        self,
        forecaster,
        cv,
        param_grid,
        scoring=None,
        strategy="refit",
        n_jobs=None,
        refit=True,
        verbose=0,
        pre_dispatch="2*n_jobs",
    ):
        super(ForecastingGridSearchCV, self).__init__(
            forecaster=forecaster,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            strategy=strategy,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
        )
        self.param_grid = param_grid

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        _check_param_grid(self.param_grid)
        return evaluate_candidates(ParameterGrid(self.param_grid))


class ForecastingRandomizedSearchCV(BaseGridSearch):
    """
    Performs randomized-search cross-validation to find optimal model parameters.
    The forecaster is fit on the initial window and then temporal
    cross-validation is used to find the optimal parameter

    Randomized cross-validation is performed based on a cross-validation
    iterator encoding the cross-validation scheme, the parameter distributions to
    search over, and (optionally) the evaluation metric for comparing model
    performance. As in scikit-learn, tuning works through the common
    hyper-parameter interface which allows to repeatedly fit and evaluate
    the same forecaster with different hyper-parameters.

    Parameters
    ----------
    forecaster : estimator object
        The estimator should implement the sktime or scikit-learn estimator
        interface. Either the estimator must contain a "score" function,
        or a scoring function must be passed.
    cv : cross-validation generator or an iterable
        e.g. SlidingWindowSplitter()
    param_distributions : dict or list of dicts
        Dictionary with parameters names (`str`) as keys and distributions
        or lists of parameters to try. Distributions must provide a ``rvs``
        method for sampling (such as those from scipy.stats.distributions).
        If a list is given, it is sampled uniformly.
        If a list of dicts is given, first a dict is sampled uniformly, and
        then a parameter is sampled using that dict as above.
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.
    scoring: function, optional (default=None)
        Function to score models for evaluation of optimal parameters
    n_jobs: int, optional (default=None)
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    refit: bool, optional (default=True)
        Refit the forecaster with the best parameters on all the data
    verbose: int, optional (default=0)
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
    pre_dispatch: str, optional (default='2*n_jobs')

    Attributes
    ----------
    best_index_ : int
    best_score_: float
        Score of the best model
    best_params_ : dict
        Best parameter values across the parameter grid
    best_forecaster_ : estimator
        Fitted estimator with the best parameters
    cv_results_ : dict
        Results from grid search cross validation
    """

    _required_parameters = ["forecaster", "cv", "param_distributions"]

    def __init__(
        self,
        forecaster,
        cv,
        param_distributions,
        n_iter=10,
        scoring=None,
        strategy="refit",
        n_jobs=None,
        refit=True,
        verbose=0,
        random_state=None,
        pre_dispatch="2*n_jobs",
    ):
        super(ForecastingRandomizedSearchCV, self).__init__(
            forecaster=forecaster,
            scoring=scoring,
            strategy=strategy,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions"""
        return evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state
            )
        )
