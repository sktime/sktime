#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__author__ = ["Markus Löning"]
__all__ = ["ForecastingGridSearchCV"]

import numbers
import time
import warnings
from collections import defaultdict
from contextlib import suppress
from functools import partial
from traceback import format_exception_only

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import check_cv
from sklearn.model_selection._search import _check_param_grid
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils.metaestimators import if_delegate_has_method

from sktime.exceptions import FitFailedWarning
from sktime.exceptions import NotFittedError
from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.base._base import DEFAULT_ALPHA
from sktime.utils.validation.forecasting import check_scoring
from sktime.utils.validation.forecasting import check_y


def _score(y_test, y_pred, scorer):
    """Evaluate forecasts"""
    if not isinstance(y_pred, pd.Series):
        raise NotImplementedError(
            "multi-step forecasting horizons with multiple cutoffs/windows "
            "are not supported yet"
        )

    # select only test points for which we have made predictions
    if not np.all(np.isin(y_pred.index, y_test.index)):
        raise IndexError("Predicted time points are not in test set")
    y_test = y_test.loc[y_pred.index]

    scores = {name: func(y_test, y_pred) for name, func in scorer.items()}
    return _check_scores(scores, scorer)


def _check_scores(scores, scorer):
    """Check returned scores"""
    error_msg = "scoring must return a number, got %s (%s) " "instead. (scorer=%s)"
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, "item"):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, "item"):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores


def _update_score(forecaster, cv, y, X, scorer):
    """Make, update and evaluate forecasts"""
    y_pred = forecaster.update_predict(y, cv=cv, X=X)
    return _score(y, y_pred, scorer)


def _split(y, X, cv):
    """Split data into training and validation window"""
    training_window, validation_window = cv.split_initial(y)
    y_train = y.iloc[training_window]
    y_val = y.iloc[validation_window]

    if X is not None:
        X_train = X.iloc[training_window, :]
        X_val = X.iloc[validation_window, :]
    else:
        X_train = None
        X_val = None

    return y_train, y_val, X_train, X_val


def _fit_and_score(
    forecaster,
    cv,
    y,
    X,
    scorer,
    verbose,
    parameters,
    fit_params,
    return_parameters=False,
    return_times=False,
    return_train_score=False,
    return_forecaster=False,
    error_score=np.nan,
):
    if return_train_score:
        raise NotImplementedError()

    # Get forecasting horizon
    fh = cv.get_fh()

    # Fit params
    fit_params = fit_params if fit_params is not None else {}
    if parameters is not None:
        forecaster.set_params(**parameters)

    # Split training data into training set and validation set
    y_train, y_val, X_train, X_val = _split(y, X, cv)

    # Fit forecaster
    start_time = time.time()
    try:
        forecaster.fit(y_train, X_train, fh)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == "raise":
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
            else:
                test_scores = error_score
            warnings.warn(
                "forecaster fit failed. The score on this train-test"
                " partition for these parameters will be set to %f. "
                "Details: \n%s" % (error_score, format_exception_only(type(e), e)[0]),
                FitFailedWarning,
            )
        else:
            raise ValueError(
                "error_score must be the string 'raise' or a"
                " numeric value. (Hint: if using 'raise', please"
                " make sure that it has been spelled correctly.)"
            )

    else:
        fit_time = time.time() - start_time
        test_scores = _update_score(forecaster, cv, y_val, X_val, scorer)
        score_time = time.time() - start_time - fit_time

    ret = [test_scores]

    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_forecaster:
        ret.append(forecaster)
    return ret


class BaseGridSearch(BaseForecaster):
    def __init__(
        self,
        forecaster,
        cv,
        n_jobs=None,
        pre_dispatch=None,
        refit=False,
        scoring=None,
        verbose=0,
        error_score=None,
        return_train_score=None,
    ):
        self.forecaster = forecaster
        self.cv = cv
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score
        super(BaseGridSearch, self).__init__()

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def update(self, y, X=None, update_params=False):
        """Call predict on the forecaster with the best found parameters."""
        self.check_is_fitted("update")
        self.best_forecaster_.update(y, X, update_params=update_params)
        return self

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def update_predict(
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
        return self.best_forecaster_.update_predict(
            y,
            cv=cv,
            X=X,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def update_predict_single(
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
        return self.best_forecaster_.update_predict_single(
            y,
            fh=fh,
            X=X,
            update_params=update_params,
            return_pred_int=return_pred_int,
            alpha=alpha,
        )

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def predict(self, fh=None, X=None, return_pred_int=False, alpha=DEFAULT_ALPHA):
        """Call predict on the forecaster with the best found parameters."""
        self.check_is_fitted("predict")
        return self.best_forecaster_.predict(
            fh, X, return_pred_int=return_pred_int, alpha=alpha
        )

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
        X : pandas.DataFrame, shape=[n_obs, n_vars], optional (default=None)
            An optional 2-d dataframe of exogenous variables.
        y : pandas.Series
            Target time series to which to compare the forecasts.
        Returns
        -------
        score : float
        """
        self.check_is_fitted("score")
        if self.scorer_ is None:
            raise ValueError(
                "No score function explicitly defined, "
                "and the forecaster doesn't provide one %s" % self.best_forecaster_
            )
        score = self.scorer_
        y_pred = self.best_forecaster_.predict(fh, X)
        return score(y, y_pred)

    def _run_search(self, evaluate_candidates):
        raise NotImplementedError("_run_search not implemented.")

    @staticmethod
    def _format_results(candidate_params, scorers, out):
        n_candidates = len(candidate_params)
        (test_score_dicts, fit_time, score_time) = zip(*out)
        test_scores = _aggregate_score_dicts(test_score_dicts)

        results = {}

        def _store(key_name, array, rank=False, greater_is_better=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64)

            results["mean_%s" % key_name] = array

            if rank:
                array = -array if greater_is_better else array
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(array, method="min"), dtype=np.int32
                )

        _store("fit_time", fit_time)
        _store("score_time", score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(
            partial(
                np.ma.MaskedArray,
                np.empty(
                    n_candidates,
                ),
                mask=True,
                dtype=object,
            )
        )
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key "params"
        results["params"] = candidate_params

        for scorer_name, scorer in scorers.items():
            # Computed the (weighted) mean and std for test scores alone
            _store(
                "test_%s" % scorer_name,
                test_scores[scorer_name],
                rank=True,
                greater_is_better=scorer.greater_is_better,
            )

        return results

    def check_is_fitted(self, method_name=None):
        super(BaseGridSearch, self).check_is_fitted()

        if method_name is not None:
            if not self.refit:
                raise NotFittedError(
                    "This %s instance was initialized "
                    "with refit=False. %s is "
                    "available only after refitting on the "
                    "best "
                    "parameters. You can refit an forecaster "
                    "manually using the ``best_params_`` "
                    "attribute" % (type(self).__name__, method_name)
                )
            else:
                self.best_forecaster_.check_is_fitted()

    def fit(self, y, X=None, fh=None, **fit_params):
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
        y = check_y(y)

        # validate cross-validator
        cv = check_cv(self.cv)
        base_forecaster = clone(self.forecaster)

        scoring = check_scoring(self.scoring)
        scorers = {scoring.name: scoring}
        refit_metric = scoring.name

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose,
        )

        results = {}
        all_candidate_params = []
        all_out = []

        def evaluate_candidates(candidate_params):
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)

            if self.verbose > 0:
                n_splits = cv.get_n_splits(y)
                print(  # noqa
                    "Fitting {0} folds for each of {1} candidates,"
                    " totalling {2} fits".format(
                        n_splits, n_candidates, n_candidates * n_splits
                    )
                )

            out = []
            for parameters in candidate_params:
                r = _fit_and_score(
                    clone(base_forecaster),
                    cv,
                    y,
                    X,
                    parameters=parameters,
                    **fit_and_score_kwargs
                )
                out.append(r)

            n_splits = cv.get_n_splits(y)

            if len(out) < 1:
                raise ValueError(
                    "No fits were performed. "
                    "Was the CV iterator empty? "
                    "Were there no candidates?"
                )

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

            nonlocal results
            results = self._format_results(all_candidate_params, scorers, all_out)
            return results

        self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_%s" % refit_metric].argmin()
        self.best_score_ = results["mean_test_%s" % refit_metric][self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        self.best_forecaster_ = clone(base_forecaster).set_params(**self.best_params_)

        if self.refit:
            refit_start_time = time.time()
            self.best_forecaster_.fit(y, X, fh)
            self.refit_time_ = time.time() - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers[scoring.name]

        self.cv_results_ = results
        self.n_splits_ = cv.get_n_splits(y)

        self._is_fitted = True
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
    """

    _required_parameters = ["forecaster", "cv", "param_grid"]

    def __init__(
        self,
        forecaster,
        cv,
        param_grid,
        scoring=None,
        n_jobs=None,
        refit=True,
        verbose=0,
        pre_dispatch="2*n_jobs",
        error_score=np.nan,
        return_train_score=False,
    ):
        super(ForecastingGridSearchCV, self).__init__(
            forecaster=forecaster,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            pre_dispatch=pre_dispatch,
            error_score=error_score,
            return_train_score=return_train_score,
        )
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))
