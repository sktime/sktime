# adapted from https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/model_selection/_search.py

__author__ = "Markus Löning"
__all__ = ["ForecastingGridSearchCV", "RollingWindowSplit"]

import numbers
import time
import warnings
from collections import defaultdict
from contextlib import suppress
from functools import partial
from itertools import product
from traceback import format_exception_only

import numpy as np
from scipy.stats import rankdata
from sklearn.base import clone
from sklearn.exceptions import FitFailedWarning
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import check_cv, ParameterGrid
from sklearn.model_selection._validation import _aggregate_score_dicts
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.utils.validation import check_is_fitted
from sktime.performance_metrics.forecasting import smape_score
from sktime.utils.validation.forecasting import check_integer_time_index
from sktime.utils.validation.forecasting import validate_fh
from sktime.utils.time_series import compute_relative_to_n_timepoints
from sktime.forecasters.base import BaseForecaster


def _score(forecaster, fh, y_test, scorer):
    """Compute the score(s) of an forecaster on a given test set.
    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    # if isinstance(scorer, dict):
    #     # will cache method calls if needed. scorer() returns a dict
    #     scorer = _MultimetricScorer(**scorer)
    # if y_test is None:
    #     scores = scorer(forecaster, X_test)
    # else:
    #     scores = scorer(forecaster, X_test, y_test)
    y_pred = forecaster.predict(fh=fh)
    scores = {name: func(y_test, y_pred) for name, func in scorer.items()}

    error_msg = ("scoring must return a number, got %s (%s) "
                 "instead. (scorer=%s)")
    if isinstance(scores, dict):
        for name, score in scores.items():
            if hasattr(score, 'item'):
                with suppress(ValueError):
                    # e.g. unwrap memmapped scalars
                    score = score.item()
            if not isinstance(score, numbers.Number):
                raise ValueError(error_msg % (score, type(score), name))
            scores[name] = score
    else:  # scalar
        if hasattr(scores, 'item'):
            with suppress(ValueError):
                # e.g. unwrap memmapped scalars
                scores = scores.item()
        if not isinstance(scores, numbers.Number):
            raise ValueError(error_msg % (scores, type(scores), scorer))
    return scores


def _fit_and_score(forecaster, y, fh, scorer, train, test, verbose,
                   parameters, fit_params, X=None,
                   return_parameters=False,
                   return_n_test_timepoints=False,
                   return_times=False,
                   return_forecaster=False,
                   error_score=np.nan):
    if verbose > 1:
        if parameters is None:
            msg = ''
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v)
                                    for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))

    # Fit params
    fit_params = fit_params if fit_params is not None else {}

    if parameters is not None:
        forecaster.set_params(**parameters)

    start_time = time.time()

    y_train = y.iloc[train]
    y_test = y.iloc[test]

    try:
        forecaster.fit(y_train, fh=fh, X=X, **fit_params)

    except Exception as e:
        # Note fit time as time until error
        fit_time = time.time() - start_time
        score_time = 0.0
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            if isinstance(scorer, dict):
                test_scores = {name: error_score for name in scorer}
                # if return_train_score:
                #     train_scores = test_scores.copy()
            else:
                test_scores = error_score
                # if return_train_score:
                #     train_scores = error_score
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
        test_scores = _score(forecaster, fh, y_test, scorer)
        score_time = time.time() - start_time - fit_time

    # if verbose > 1:
    #     total_time = score_time + fit_time
    #     print(_message_with_time('CV', msg, total_time))

    ret = [test_scores]

    if return_n_test_timepoints:
        ret.append(len(test))
    if return_times:
        ret.extend([fit_time, score_time])
    if return_parameters:
        ret.append(parameters)
    if return_forecaster:
        ret.append(forecaster)
    return ret


class ForecastingGridSearchCV(BaseForecaster):

    def __init__(self, forecaster, param_grid, cv, refit=False, scoring=None, verbose=0):
        self.forecaster = forecaster
        self.param_grid = param_grid
        self.cv = cv
        self.refit = refit
        self.scoring = scoring if scoring is not None else smape_score
        self.verbose = verbose
        super(ForecastingGridSearchCV, self).__init__()

    def _fit(self, y, fh=None, X=None, **fit_params):
        """Internal fit"""

        # validate cross-validator
        cv = check_cv(self.cv)

        # get integer time index
        time_index = check_integer_time_index(self._time_index)

        base_forecaster = clone(self.forecaster)

        # scorers, self.multimetric_ = _check_multimetric_scoring(
        #     self.forecaster, scoring=self.scoring)
        scorers = {"score": self.scoring}
        refit_metric = "score"

        # if self.multimetric_:
        #     if self.refit is not False and (
        #             not isinstance(self.refit, str) or
        #             # This will work for both dict / list (tuple)
        #             self.refit not in scorers) and not callable(self.refit):
        #         raise ValueError("For multi-metric scoring, the parameter "
        #                          "refit must be set to a scorer key or a "
        #                          "callable to refit an forecaster with the "
        #                          "best parameter setting on the whole "
        #                          "data and make the best_* attributes "
        #                          "available for that metric. If this is "
        #                          "not needed, refit should be set to "
        #                          "False explicitly. %r was passed."
        #                          % self.refit)
        #     else:
        #         refit_metric = self.refit
        # else:
        #     refit_metric = 'score'

        fit_and_score_kwargs = dict(
            scorer=scorers,
            fit_params=fit_params,
            # return_train_score=self.return_train_score,
            return_n_test_timepoints=True,
            return_times=True,
            return_parameters=False,
            # error_score=self.error_score,
            verbose=self.verbose
        )

        results = {}
        all_candidate_params = []
        all_out = []

        def evaluate_candidates(candidate_params):
            candidate_params = list(candidate_params)
            n_candidates = len(candidate_params)

            # if self.verbose > 0:
            #     print("Fitting {0} folds for each of {1} candidates,"
            #           " totalling {2} fits".format(
            #         n_splits, n_candidates, n_candidates * n_splits))

            out = []
            for parameters, (train, test) in product(candidate_params, cv.split(time_index)):
                r = _fit_and_score(
                    clone(base_forecaster),
                    y,
                    fh,
                    X=X,
                    train=train,
                    test=test,
                    parameters=parameters,
                    **fit_and_score_kwargs
                )
                out.append(r)

            n_splits = cv.get_n_splits()

            if len(out) < 1:
                raise ValueError('No fits were performed. '
                                 'Was the CV iterator empty? '
                                 'Were there no candidates?')
            elif len(out) != n_candidates * n_splits:
                raise ValueError('cv.split and cv.get_n_splits returned '
                                 'inconsistent results. Expected {} '
                                 'splits, got {}'
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
            self.best_forecaster_.fit(y, fh=fh, **fit_params)
            refit_end_time = time.time()
            self.refit_time_ = refit_end_time - refit_start_time

        # Store the only scorer not as a dict for single metric evaluation
        self.scorer_ = scorers['score']

        self.cv_results_ = results
        self.n_splits_ = cv.get_n_splits()

        return self

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid"""
        evaluate_candidates(ParameterGrid(self.param_grid))

    def _format_results(self, candidate_params, scorers, n_splits, out):
        n_candidates = len(candidate_params)

        # if one choose to see train score, "out" will contain train score info
        # if self.return_train_score:
        #     (train_score_dicts, test_score_dicts, test_sample_counts, fit_time,
        #      score_time) = zip(*out)
        # else:
        #     (test_score_dicts, test_sample_counts, fit_time,
        #      score_time) = zip(*out)
        (test_score_dicts, test_sample_counts, fit_time,
         score_time) = zip(*out)

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
            results['mean_%s' % key_name] = array_means
            # Weighted std is not directly available in numpy
            array_stds = np.sqrt(np.average((array -
                                             array_means[:, np.newaxis]) ** 2,
                                            axis=1, weights=weights))
            results['std_%s' % key_name] = array_stds

            if rank:
                results["rank_%s" % key_name] = np.asarray(
                    rankdata(-array_means, method='min'), dtype=np.int32)

        _store('fit_time', fit_time)
        _store('score_time', score_time)
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
        # Store a list of param dicts at the key 'params'
        results['params'] = candidate_params

        for scorer_name in scorers.keys():
            # Computed the (weighted) mean and std for test scores alone
            _store('test_%s' % scorer_name, test_scores[scorer_name],
                   splits=True, rank=True,
                   weights=None)

        return results

    def _check_is_fitted(self, method_name):
        if not self.refit:
            raise NotFittedError('This %s instance was initialized '
                                 'with refit=False. %s is '
                                 'available only after refitting on the best '
                                 'parameters. You can refit an forecaster '
                                 'manually using the ``best_params_`` '
                                 'attribute'
                                 % (type(self).__name__, method_name))
        else:
            check_is_fitted(self, "best_forecaster_")

    @if_delegate_has_method(delegate=('best_forecaster_', 'forecaster'))
    def _predict(self, fh=None, X=None):
        """Call predict on the forecaster with the best found parameters.
        """
        self._check_is_fitted('predict')
        return self.best_forecaster_.predict(fh=fh, X=X)

    @if_delegate_has_method(delegate=('best_forecaster_', 'forecaster'))
    def transform(self, y):
        """Call transform on the forecaster with the best found parameters.
        """
        self._check_is_fitted('transform')
        return self.best_forecaster_.transform(y)

    @if_delegate_has_method(delegate=('best_forecaster_', 'forecaster'))
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
        self._check_is_fitted('inverse_transform')
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
        self._check_is_fitted('score')
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the forecaster doesn't provide one %s"
                             % self.best_forecaster_)
        score = self.scorer_
        y_pred = self.best_forecaster_.predict(fh, X=X)
        return score(y_true, y_pred)


class RollingWindowSplit:
    """Rolling window iterator that allows to split time series index into two windows,
    one containing observations used as feature data and one containing observations used as
    target data to be predicted. The target window has the length of the given forecasting horizon.

    Parameters
    ----------
    window_length : int, optional (default is sqrt of time series length)
        Length of rolling window
    fh : array-like  or int, optional, (default=None)
        Single step ahead or array of steps ahead to forecast.
    """

    def __init__(self, fh, window_length=None, n_splits=None):
        # check input
        self.fh = validate_fh(fh)

        # either window_length or n_splits must be specified, but not both
        if ((window_length is None) and (n_splits is None)) \
                or ((window_length is not None) and (n_splits is not None)):
            raise ValueError("Either `window_length` or `n_splits` must be specified but not both.")
        self.window_length = window_length
        self.n_splits = n_splits

        # computed when calling split
        self.n_splits_ = None
        self.window_length_ = None

    def split(self, y):
        """
        Split data using rolling window.

        Parameters
        ----------
        y : ndarray
            1-dimensional array of time series index to split.

        Yields
        ------
        y_input : ndarray, shape=(window_length,)
            The indices of the feature window
        y_output : ndarray, shape=(len(fh),)
            The indices of the target window
        """

        # check input
        if not isinstance(y, np.ndarray):
            raise ValueError(f"`y` has to be numpy array, but found: {type(y)}")
        if y.ndim != 1:
            raise ValueError(f"`y` has to be 1d array")

        n_timepoints = len(y)
        fh_max = self.fh[-1]  # furthest step ahead, assume fh is sorted
        last_window_end = n_timepoints - fh_max + 1

        # compute missing window_length/n_splits depending on what was passed in the constructor;
        # window_length/n_splits can be specified relative to length of passed time series y using functions,
        # floats or kwargs, so first compute the actual value, before computing the missing value
        if self.window_length is not None:
            window_length = compute_relative_to_n_timepoints(n_timepoints, self.window_length)
            n_splits = last_window_end - window_length
        else:
            n_splits = compute_relative_to_n_timepoints(n_timepoints, self.n_splits)
            window_length = last_window_end - n_splits

        self.n_splits_ = n_splits
        self.window_length_ = window_length

        # check if computed values are feasible given n_timepoints
        if window_length + fh_max > n_timepoints:
            raise ValueError(f"`window_length` + `max(fh)` must be smaller than "
                             f"the number of time points in `y`, but found: "
                             f"{window_length} + {fh_max} > {n_timepoints}")

        # iterate over windows
        start = window_length
        for window in range(start, last_window_end):
            y_input = y[window - window_length:window]
            y_output = y[window + self.fh - 1]
            yield y_input, y_output

    def get_n_splits(self):
        """
        Return number of splits.
        """
        if self.n_splits_ is None:
            raise ValueError(f"`n_splits_` is only available after calling `split`. "
                             f"This is because it depends on the number of time points of the "
                             f"time series `y` which is passed to split.")
        else:
            return self.n_splits_

    def get_window_length(self):
        """
        Return the window length.
        """
        if self.window_length_ is None:
            raise ValueError(f"`window_length_` is only available after calling `split`. "
                             f"This is because it depends on the number of time points of the "
                             f"time series `y` which is passed to split.")
        return self.window_length_


def split_into_tabular_train_test(x, window_length=None, fh=None, test_size=1):
    """Helper function to split single time series into tabular train and
    test sets using rolling window approach"""

    # validate forecasting horizon
    fh = validate_fh(fh)

    # get time series index
    index = np.arange(len(x))

    # set up rolling window iterator
    rw = RollingWindowSplit(window_length=window_length, fh=fh)

    # slice time series into windows
    xs = []
    ys = []
    for input, output in rw.split(index):
        xt = x[input]
        yt = x[output]
        xs.append(xt)
        ys.append(yt)

    # stack windows into tabular array
    x = np.array(xs)
    y = np.array(ys)

    # split into train and test set
    x_train = x[:-test_size, :]
    y_train = y[:-test_size, :]

    x_test = x[-test_size:, :]
    y_test = y[-test_size:, :]

    return x_train, y_train, x_test, y_test


def temporal_train_test_split(y, fh):
    """Split single time series into train and test at given forecasting horizon"""
    fh = validate_fh(fh)
    fh_max = np.max(fh)
    fh_idx = fh - 1  # zero indexing

    y_train = y.iloc[:-fh_max]
    y_test = y.iloc[-fh_max:].iloc[fh_idx]

    return y_train, y_test
