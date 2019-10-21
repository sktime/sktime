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

from sktime.utils.data_container import select_times, get_time_index
from sktime.utils.validation.forecasting import validate_fh, validate_y_X
from sktime.utils.time_series import compute_n_intervals


def _score(estimator, fh, y_test, scorer):
    """Compute the score(s) of an estimator on a given test set.
    Will return a dict of floats if `scorer` is a dict, otherwise a single
    float is returned.
    """
    # if isinstance(scorer, dict):
    #     # will cache method calls if needed. scorer() returns a dict
    #     scorer = _MultimetricScorer(**scorer)
    # if y_test is None:
    #     scores = scorer(estimator, X_test)
    # else:
    #     scores = scorer(estimator, X_test, y_test)
    y_pred = estimator.predict(fh=fh)
    scores = {name: func(y_test.iloc[0], y_pred) for name, func in scorer.items()}

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


def _fit_and_score(estimator, y, fh, scorer, train, test, verbose,
                   parameters, fit_params, X=None,
                   return_parameters=False,
                   return_n_test_timepoints=False,
                   return_times=False,
                   return_estimator=False,
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
        estimator.set_params(**parameters)

    start_time = time.time()

    y_train = select_times(y, train)
    y_test = select_times(y, test)

    try:
        estimator.fit(y_train, fh=fh, **fit_params)

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
            warnings.warn("estimator fit failed. The score on this train-test"
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
        test_scores = _score(estimator, fh, y_test, scorer)
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
    if return_estimator:
        ret.append(estimator)
    return ret


class ForecastingGridSearchCV:

    def __init__(self, estimator, param_grid, cv, refit=False, scoring=None, verbose=0, check_input=True):
        self.estimator = estimator
        self.param_grid = param_grid
        self.cv = cv
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.check_input = check_input

    def fit(self, y, fh=None, X=None, **fit_params):

        if self.check_input:
            validate_y_X(y, X)

        # validate forecasting horizon
        if fh is not None:
            fh = validate_fh(fh)

        # validate cross-validator
        cv = check_cv(self.cv)

        # Keep index for predicting where forecasters horizon will be relative to y seen in fit
        self._time_index = get_time_index(y)

        # Make interface compatible with estimators that only take y and no X
        if X is not None:
            raise NotImplementedError("Exogeneous variables not supported yet")
        # kwargs = {} if X is None else {'X': X}

        time_index = self._time_index.values

        base_estimator = clone(self.estimator)

        # scorers, self.multimetric_ = _check_multimetric_scoring(
        #     self.estimator, scoring=self.scoring)
        scorers = {"score": self.scoring}
        refit_metric = "score"

        # if self.multimetric_:
        #     if self.refit is not False and (
        #             not isinstance(self.refit, str) or
        #             # This will work for both dict / list (tuple)
        #             self.refit not in scorers) and not callable(self.refit):
        #         raise ValueError("For multi-metric scoring, the parameter "
        #                          "refit must be set to a scorer key or a "
        #                          "callable to refit an estimator with the "
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
                    clone(base_estimator),
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

        self.best_estimator_ = clone(base_estimator).set_params(
            **self.best_params_)

        if self.refit:
            refit_start_time = time.time()
            self.best_estimator_.fit(y, fh, **fit_params)
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
                                 'parameters. You can refit an estimator '
                                 'manually using the ``best_params_`` '
                                 'attribute'
                                 % (type(self).__name__, method_name))
        else:
            check_is_fitted(self, "best_estimator_")

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def predict(self, fh=None):
        """Call predict on the estimator with the best found parameters.

        Only available if ``refit=True`` and the underlying estimator supports
        ``predict``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('predict')
        return self.best_estimator_.predict(fh=fh)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def transform(self, X):
        """Call transform on the estimator with the best found parameters.

        Only available if the underlying estimator supports ``transform`` and
        ``refit=True``.

        Parameters
        ----------
        X : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('transform')
        return self.best_estimator_.transform(X)

    @if_delegate_has_method(delegate=('best_estimator_', 'estimator'))
    def inverse_transform(self, Xt):
        """Call inverse_transform on the estimator with the best found params.

        Only available if the underlying estimator implements
        ``inverse_transform`` and ``refit=True``.

        Parameters
        ----------
        Xt : indexable, length n_samples
            Must fulfill the input assumptions of the
            underlying estimator.

        """
        self._check_is_fitted('inverse_transform')
        return self.best_estimator_.inverse_transform(Xt)

    def score(self, y, fh, X=None):
        """Returns the score on the given data, if the estimator has been refit.

        This uses the score defined by ``scoring`` where provided, and the
        ``best_estimator_.score`` method otherwise.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Input data, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples] or [n_samples, n_output], optional
            Target relative to X for classification or regression;
            None for unsupervised learning.

        Returns
        -------
        score : float
        """
        self._check_is_fitted('score')
        if self.scorer_ is None:
            raise ValueError("No score function explicitly defined, "
                             "and the estimator doesn't provide one %s"
                             % self.best_estimator_)
        score = self.scorer_
        y_pred = self.best_estimator_.predict(fh, X=X)
        return score(y, y_pred)


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

    def __init__(self, window_length="log", fh=1):

        self.window_length = window_length
        self.fh = validate_fh(fh)

        # Attributes updated in split
        self.n_splits_ = None
        self.window_length_ = None

    def split(self, data):
        """
        Split data using rolling window.

        Parameters
        ----------
        data : ndarray
            1-dimensional array of time series index to split.

        Yields
        ------
        features : ndarray
            The indices of the feature window
        targets : ndarray
            The indices of the target window
        """

        # Input checks.
        if not (isinstance(data, np.ndarray) and data.ndim == 1):
            raise ValueError(f"Passed data has to be 1-d numpy array, but found "
                             f"data of type: {type(data)} with {data.ndim} dimensions")

        n_timepoints = data.shape[0]
        max_fh = self.fh[-1]  # furthest step ahead, assume fh is sorted

        # Set default window length to sqrt of series length
        window_length = compute_n_intervals(n_timepoints, self.window_length)

        # Iterate over windows
        start = window_length
        stop = n_timepoints - max_fh + 1
        self.n_splits_ = stop - start

        for window in range(start, stop):
            inputs = data[window - window_length:window]
            outputs = data[window + self.fh - 1]
            yield inputs, outputs

    def get_n_splits(self):
        """
        Return number of splits.
        """
        return self.n_splits_

    def get_window_length(self):
        """
        Return the window length.
        """
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