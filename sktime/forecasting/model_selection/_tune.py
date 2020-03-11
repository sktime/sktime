#!/usr/bin/env python3 -u
# coding: utf-8

__author__ = ["Markus Löning"]
__all__ = ["ForecastingGridSearchCV"]

import numbers
import time
import warnings
from collections import defaultdict
from collections.abc import Sequence
from contextlib import suppress
from functools import partial
from traceback import format_exception_only

import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import check_cv, ParameterGrid
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils.metaestimators import if_delegate_has_method
from sktime.forecasting.base.base import BaseForecaster
from sktime.forecasting.base.base import DEFAULT_ALPHA
from sktime.utils.validation.forecasting import check_cv
from sktime.utils.validation.forecasting import check_y


def _check_param_grid(param_grid):
    if hasattr(param_grid, 'items'):
        param_grid = [param_grid]

    for p in param_grid:
        for name, v in p.items():
            if isinstance(v, np.ndarray) and v.ndim > 1:
                raise ValueError("Parameter array should be one-dimensional.")

            if (isinstance(v, str) or
                    not isinstance(v, (np.ndarray, Sequence))):
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a sequence(but not a string) or"
                                 " np.ndarray.".format(name))

            if len(v) == 0:
                raise ValueError("Parameter values for parameter ({0}) need "
                                 "to be a non-empty sequence.".format(name))


def _score(y_test, y_pred, scorer):
    scores = {name: func(y_test, y_pred) for name, func in scorer.items()}

    error_msg = ("scoring must return a number, got %s (%s) "
                 "instead. (scorer=%s)")
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


def _update_score(forecaster, cv, y_test, X_test, scorer):
    """Compute the score(s) of an forecaster on a given test set.
    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    y_pred = forecaster.update_predict(y_test, cv=cv, X_test=X_test)
    return _score(y_test, y_pred, scorer)


def _fit_and_score(forecaster, cv, y, X, scorer, verbose,
                   parameters, fit_params,
                   return_parameters=False,
                   return_times=False,
                   return_train_score=False,
                   return_forecaster=False,
                   error_score=np.nan):
    if return_train_score:
        raise NotImplementedError()

    # Get forecasting horizon
    fh = cv.get_fh()

    # Fit params
    fit_params = fit_params if fit_params is not None else {}
    if parameters is not None:
        forecaster.set_params(**parameters)

    # Split training data into training set and validation set
    training_window, _ = cv.split_initial(y)
    y_train = y.iloc[training_window]
    X_train = X.iloc[training_window, :] if X is not None else None

    # Fit forecaster
    start_time = time.time()
    try:
        forecaster.fit(y_train, fh, X_train=X_train, **fit_params)

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
            warnings.warn("forecaster fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%s" %
                          (error_score, format_exception_only(type(e), e)[0]),
                          FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)")

    else:
        fit_time = time.time() - start_time
        test_scores = _update_score(forecaster, cv, y, X, scorer)
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

    def __init__(self, forecaster, cv, n_jobs=None, pre_dispatch=None, refit=False, scoring=None, verbose=0,
                 error_score=None, return_train_score=None):
        self.forecaster = forecaster
        self.cv = cv
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.error_score = error_score
        self.return_train_score = return_train_score
        self.best_forecaster_ = None
        super(BaseGridSearch, self).__init__()

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def update(self, y_new, X_new=None, update_params=False):
        """Call predict on the forecaster with the best found parameters.
        """
        self._check_is_fitted("update")
        return self.best_forecaster_.update(y_new, X_new=X_new, update_params=update_params)

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def update_predict(self, y_test, cv=None, X_test=None, update_params=False, return_pred_int=False,
                       alpha=DEFAULT_ALPHA):
        """Call predict on the forecaster with the best found parameters.
        """
        self._check_is_fitted("update_predict")
        return self.best_forecaster_.update_predict(y_test, cv=cv, X_test=X_test, update_params=update_params,
                                                    return_pred_int=return_pred_int, alpha=alpha)

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def update_predict_single(self, y_new, fh=None, X=None, update_params=False, return_pred_int=False,
                              alpha=DEFAULT_ALPHA):
        """Call predict on the forecaster with the best found parameters.
        """
        self._check_is_fitted("update_predict_single")
        return self.best_forecaster_.update_predict_single(y_new, fh=fh, X=X, update_params=update_params,
                                                           return_pred_int=return_pred_int, alpha=alpha)

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def predict(self, fh=None, X=None):
        """Call predict on the forecaster with the best found parameters.
        """
        self._check_is_fitted("predict")
        return self.best_forecaster_.predict(fh=fh, X=X)

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def transform(self, y, **transform_params):
        self._check_is_fitted("transform")
        return self.best_forecaster_.transform(y, **transform_params)

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def get_fitted_params(self, y):
        """Call transform on the forecaster with the best found parameters.
        """
        self._check_is_fitted("transform")
        return self.best_forecaster_.get_fitted_params()

    @if_delegate_has_method(delegate=("best_forecaster_", "forecaster"))
    def inverse_transform(self, y):
        """Call inverse_transform on the forecaster with the best found params.
        Only available if the underlying forecaster implements
        ``inverse_transform`` and ``refit=True``.
        Parameters
        ----------
        y : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying forecaster.
        """
        self._check_is_fitted("inverse_transform")
        return self.best_forecaster_.inverse_transform(y)

    def score(self, y_true, fh=None, X=None):
        """Returns the score on the given data, if the forecaster has been refit.
        This uses the score defined by ``scoring`` where provided, and the
        ``best_forecaster_.score`` method otherwise.
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.
        y_true : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.
        Returns
        -------
        score : float
        """
        self._check_is_fitted("score")
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the forecaster doesn't provide one %s"
                             % self.best_forecaster_)
        score = self.scorer_
        y_pred = self.best_forecaster_.predict(fh, X=X)
        return score(y_true, y_pred)

    def _run_search(self, evaluate_candidates):
        raise NotImplementedError("_run_search not implemented.")

    def _format_results(self, candidate_params, scorers, n_splits, out):
        n_candidates = len(candidate_params)

        # if one choose to see train score, "out" will contain train score info
        # if self.return_train_score:
        #     (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
        #      score_time) = zip(*out)
        # else:
        #     (test_score_dicts, test_sample_counts, fit_time,
        #      score_time) = zip(*out)
        (test_score_dicts, fit_time, score_time) = zip(*out)

        # test_score_dicts and train_score dicts are lists of dictionaries and
        # we make them into dict of lists
        test_scores = _aggregate_score_dicts(test_score_dicts)
        # if self.return_train_score:
        #     train_scores = _aggregate_score_dicts(train_score_dicts)

        results = {}

        def _store(key_name, array, weights=None, splits=False, rank=False):
            """A small helper to store the scores/times to the cv_results_"""
            # When iterated first by splits, then by parameters
            # We want `array` to have `n_candidates` rows and `n_splits` cols.
            array = np.array(array, dtype=np.float64).reshape(n_candidates,
                                                              n_splits)
            if splits:
                for split_i in range(n_splits):
                    # Uses closure to alter the results
                    results["split%d_%s"
                            % (split_i, key_name)] = array[:, split_i]

            array_means = np.average(array, axis=1, weights=weights)
            results["mean_%s" % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results["std_%s" % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method="min"), dtype=np.int32)

        _store("fit_time", fit_time)
        _store("score_time", score_time)
        # Use one MaskedArray and mask all the places where the param is not
        # applicable for that candidate. Use defaultdict as each candidate may
        # not contain all the params
        param_results = defaultdict(partial(np.ma.MaskedArray,
                                            np.empty(n_candidates, ),
                                            mask=True,
                                            dtype=object))
        for cand_i, params in enumerate(candidate_params):
            for name, value in params.items():
                # An all masked empty array gets created for the key
                # `"param_%s" % name` at the first occurrence of `name`.
                # Setting the value at an index also unmasks that index
                param_results["param_%s" % name][cand_i] = value

        results.update(param_results)
        # Store a list of param dicts at the key "params"
        results["params"] = candidate_params

        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store("test_%s" % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=None)

        return results

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError("This %s instance was initialized "
                                 "with refit=False. %s is "
                                 "available only after refitting on the best "
                                 "parameters. You can refit an forecaster "
                                 "manually using the ``best_params_`` "
                                 "attribute"
                                 % (type(self).__name__, method_name))
        else:
            self.best_forecaster_._check_is_fitted()

    def fit(self, y_train, fh=None, X_train=None, **fit_params):
        """Internal fit"""
        y_train = check_y(y_train)

        # validate cross-validator
        cv = check_cv(self.cv)

        base_forecaster = clone(self.forecaster)

        if not callable(self.scoring) or self.scoring is None:
            raise NotImplementedError()
        scorers = {"score": self.scoring}
        refit_metric = "score"

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            return_train_score=self.return_train_score,
            return_times=True,
            return_parameters=False,
            error_score=self.error_score,
            verbose=self.verbose
        )

        results = {}
        all_candidate_params = []
        all_out = []

        def evaluate_candidates(candidate_params):
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)

            if self.verbose > 0:
                n_splits = cv.get_n_splits(y_train)
                print("Fitting {0} folds for each of {1} candidates,"
                      " totalling {2} fits".format(
                    n_splits, n_candidates, n_candidates * n_splits))

            out = []
            for parameters in candidate_params:
                r = _fit_and_score(
                    clone(base_forecaster),
                    cv,
                    y_train,
                    X_train,
                    parameters=parameters,
                    **fit_and_score_kwargs
                )
                out.append(r)

            n_splits = cv.get_n_splits(y_train)

            if len(out) < 1:
                raise ValueError("No fits were performed. "
                                 "Was the CV iterator empty? "
                                 "Were there no candidates?")
            elif len(out) != n_candidates * n_splits:
                raise ValueError("cv.split and cv.get_n_splits returned "
                                 "inconsistent results. Expected {} "
                                 "splits, got {}"
                                 .format(n_splits,
                                         len(out) // n_candidates))

            all_candidate_params.extend(candidate_params)
            all_out.extend(out)

            nonlocal results
            results = self._format_results(
                all_candidate_params, scorers, n_splits, all_out)
            return results

        self._run_search(evaluate_candidates)

        self.best_index_ = results["rank_test_%s"
                                   % refit_metric].argmin()
        self.best_score_ = results["mean_test_%s" % refit_metric][
            self.best_index_]
        self.best_params_ = results["params"][self.best_index_]

        self.best_forecaster_ = clone(base_forecaster).set_params(**self.best_params_)

        if self.refit:
            refit_start_time = time.time()
            self.best_forecaster_.fit(y_train, fh=fh, X_train=X_train, **fit_params)
            self.refit_time_ = time.time() - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers["score"]

        self.cv_results_ = results
        self.n_splits_ = cv.get_n_splits()

        return self


class ForecastingGridSearchCV(BaseGridSearch):

    def __init__(self, forecaster, param_grid, scoring=None,
                 n_jobs=None, refit=True, cv=None,
                 verbose=0, pre_dispatch='2*n_jobs',
                 error_score=np.nan, return_train_score=False):
        super(ForecastingGridSearchCV, self).__init__(
            forecaster=forecaster, scoring=scoring,
            n_jobs=n_jobs, refit=refit, cv=cv, verbose=verbose,
            pre_dispatch=pre_dispatch, error_score=error_score,
            return_train_score=return_train_score)
        self.param_grid = param_grid
        _check_param_grid(param_grid)

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))
