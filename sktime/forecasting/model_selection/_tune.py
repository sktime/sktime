#!/usr/bin/env python3 -u
# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements grid search functionality to tune forecasters."""

__author__ = ["mloning"]
__all__ = ["ForecastingGridSearchCV", "ForecastingRandomizedSearchCV"]

from collections.abc import Sequence

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, ParameterSampler, check_cv

from sktime.datatypes import mtype_to_scitype
from sktime.exceptions import NotFittedError
from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.utils.validation.forecasting import check_scoring


class BaseGridSearch(_DelegatedForecaster):

    _tags = {
        "scitype:y": "both",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
    }

    def __init__(
        self,
        forecaster,
        cv,
        strategy="refit",
        n_jobs=None,
        pre_dispatch=None,
        backend="loky",
        refit=False,
        scoring=None,
        verbose=0,
        return_n_best_forecasters=1,
        update_behaviour="full_refit",
        error_score=np.nan,
    ):

        self.forecaster = forecaster
        self.cv = cv
        self.strategy = strategy
        self.n_jobs = n_jobs
        self.pre_dispatch = pre_dispatch
        self.backend = backend
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.return_n_best_forecasters = return_n_best_forecasters
        self.update_behaviour = update_behaviour
        self.error_score = error_score
        super(BaseGridSearch, self).__init__()
        tags_to_clone = [
            "requires-fh-in-fit",
            "capability:pred_int",
            "scitype:y",
            "ignores-exogeneous-X",
            "handles-missing-data",
            "y_inner_mtype",
            "X_inner_mtype",
            "X-y-must-have-same-index",
            "enforce_index_type",
        ]
        self.clone_tags(forecaster, tags_to_clone)
        self._extend_to_all_scitypes("y_inner_mtype")
        self._extend_to_all_scitypes("X_inner_mtype")

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "best_forecaster_"

    def _extend_to_all_scitypes(self, tagname):
        """Ensure mtypes for all scitypes are in the tag with tagname.

        Mutates self tag with name `tagname`.
        If no mtypes are present of a time series scitype, adds a pandas based one.

        Parameters
        ----------
        tagname : str, name of the tag. Should be "y_inner_mtype" or "X_inner_mtype".

        Returns
        -------
        None (mutates tag in self)
        """
        tagval = self.get_tag(tagname)
        if not isinstance(tagval, list):
            tagval = [tagval]
        scitypes = mtype_to_scitype(tagval, return_unique=True)
        if "Series" not in scitypes:
            tagval = tagval + ["pd.DataFrame"]
        if "Panel" not in scitypes:
            tagval = tagval + ["pd-multiindex"]
        if "Hierarchical" not in scitypes:
            tagval = tagval + ["pd_multiindex_hier"]
        self.set_tags(**{tagname: tagval})

    def _get_fitted_params(self):
        """Get fitted parameters.

        Returns
        -------
        fitted_params : dict
            A dict containing the best hyper parameters and the parameters of
            the best estimator (if available), merged together with the former
            taking precedence.
        """
        fitted_params = {}
        try:
            fitted_params = self.best_forecaster_.get_fitted_params()
        except NotImplementedError:
            pass
        fitted_params = {**fitted_params, **self.best_params_}
        return fitted_params

    def _run_search(self, evaluate_candidates):
        raise NotImplementedError("abstract method")

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
        cv = check_cv(self.cv)

        scoring = check_scoring(self.scoring)
        scoring_name = f"test_{scoring.name}"

        parallel = Parallel(
            n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch, backend=self.backend
        )

        def _fit_and_score(params):
            # Clone forecaster.
            forecaster = self.forecaster.clone()

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
                error_score=self.error_score,
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
            ascending=scoring.get_tag("lower_is_better")
        )

        self.cv_results_ = results

        # Select best parameters.
        self.best_index_ = results.loc[:, f"rank_{scoring_name}"].argmin()
        # Raise error if all fits in evaluate failed because all score values are NaN.
        if self.best_index_ == -1:
            raise NotFittedError(
                f"""All fits of forecaster failed,
                set error_score='raise' to see the exceptions.
                Failed forecaster: {self.forecaster}"""
            )
        self.best_score_ = results.loc[self.best_index_, f"mean_{scoring_name}"]
        self.best_params_ = results.loc[self.best_index_, "params"]
        self.best_forecaster_ = self.forecaster.clone().set_params(**self.best_params_)

        # Refit model with best parameters.
        if self.refit:
            self.best_forecaster_.fit(y, X, fh)

        # Sort values according to rank
        results = results.sort_values(
            by=f"rank_{scoring_name}", ascending=scoring.get_tag("lower_is_better")
        )
        # Select n best forecaster
        self.n_best_forecasters_ = []
        self.n_best_scores_ = []
        for i in range(self.return_n_best_forecasters):
            params = results["params"].iloc[i]
            rank = results[f"rank_{scoring_name}"].iloc[i]
            rank = str(int(rank))
            forecaster = self.forecaster.clone().set_params(**params)
            # Refit model with best parameters.
            if self.refit:
                forecaster.fit(y, X, fh)
            self.n_best_forecasters_.append((rank, forecaster))
            # Save score
            score = results[f"mean_{scoring_name}"].iloc[i]
            self.n_best_scores_.append(score)

        return self

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        Parameters
        ----------
        y : guaranteed to be of a type in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        update_behaviour = self.update_behaviour

        if update_behaviour == "full_refit":
            super()._update(y=y, X=X, update_params=update_params)
        elif update_behaviour == "inner_only":
            self.best_forecaster_.update(y=y, X=X, update_params=update_params)
        elif update_behaviour == "no_update":
            self.best_forecaster_.update(y=y, X=X, update_params=False)
        else:
            raise ValueError(
                'update_behaviour must be one of "full_refit", "inner_only",'
                f' or "no_update", but found {update_behaviour}'
            )
        return self


class ForecastingGridSearchCV(BaseGridSearch):
    """Perform grid-search cross-validation to find optimal model parameters.

    The forecaster is fit on the initial window and then temporal
    cross-validation is used to find the optimal parameter.

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
    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        data ingestion strategy in fitting cv, passed to `evaluate` internally
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = forecaster is refitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update
    update_behaviour: str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update
        "full_refit" = both tuning parameters and inner estimator refit on all data seen
        "inner_only" = tuning parameters are not re-tuned, inner estimator is updated
        "no_update" = neither tuning parameters nor inner estimator are updated
    param_grid : dict or list of dictionaries
        Model tuning parameters of the forecaster to evaluate
    scoring: function, optional (default=None)
        Function to score models for evaluation of optimal parameters
    n_jobs: int, optional (default=None)
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    refit: bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = best forecaster remains fitted on the last fold in cv
    verbose: int, optional (default=0)
    return_n_best_forecasters: int, default=1
        In case the n best forecaster should be returned, this value can be set
        and the n best forecasters will be assigned to n_best_forecasters_
    pre_dispatch: str, optional (default='2*n_jobs')
    error_score: numeric value or the str 'raise', optional (default=np.nan)
        The test score returned when a forecaster fails to be fitted.
    return_train_score: bool, optional (default=False)
    backend: str, optional (default="loky")
        Specify the parallelisation backend implementation in joblib, where
        "loky" is used by default.
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

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
        Number of splits in the data for cross validation
    refit_time_ : float
        Time (seconds) to refit the best forecaster
    scorer_ : function
        Function used to score model
    n_best_forecasters_: list of tuples ("rank", <forecaster>)
        The "rank" is in relation to best_forecaster_
    n_best_scores_: list of float
        The scores of n_best_forecasters_ sorted from best to worst
        score of forecasters

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.model_selection import (
    ...     ExpandingWindowSplitter,
    ...     ForecastingGridSearchCV,
    ...     ExpandingWindowSplitter)
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> y = load_shampoo_sales()
    >>> fh = [1,2,3]
    >>> cv = ExpandingWindowSplitter(fh=fh)
    >>> forecaster = NaiveForecaster()
    >>> param_grid = {"strategy" : ["last", "mean", "drift"]}
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=forecaster,
    ...     param_grid=param_grid,
    ...     cv=cv)
    >>> gscv.fit(y)
    ForecastingGridSearchCV(...)
    >>> y_pred = gscv.predict(fh)

        Advanced model meta-tuning (model selection) with multiple forecasters
        together with hyper-parametertuning at same time using sklearn notation:
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.exp_smoothing import ExponentialSmoothing
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.model_selection import ExpandingWindowSplitter
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.forecasting.theta import ThetaForecaster
    >>> from sktime.transformations.series.impute import Imputer
    >>> y = load_shampoo_sales()
    >>> pipe = TransformedTargetForecaster(steps=[
    ...     ("imputer", Imputer()),
    ...     ("forecaster", NaiveForecaster())])
    >>> cv = ExpandingWindowSplitter(
    ...     initial_window=24,
    ...     step_length=12,
    ...     fh=[1,2,3])
    >>> gscv = ForecastingGridSearchCV(
    ...     forecaster=pipe,
    ...     param_grid=[{
    ...         "forecaster": [NaiveForecaster(sp=12)],
    ...         "forecaster__strategy": ["drift", "last", "mean"],
    ...     },
    ...     {
    ...         "imputer__method": ["mean", "drift"],
    ...         "forecaster": [ThetaForecaster(sp=12)],
    ...     },
    ...     {
    ...         "imputer__method": ["mean", "median"],
    ...         "forecaster": [ExponentialSmoothing(sp=12)],
    ...         "forecaster__trend": ["add", "mul"],
    ...     },
    ...     ],
    ...     cv=cv,
    ...     n_jobs=-1)  # doctest: +SKIP
    >>> gscv.fit(y)  # doctest: +SKIP
    ForecastingGridSearchCV(...)
    >>> y_pred = gscv.predict(fh=[1,2,3])  # doctest: +SKIP
    """

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
        return_n_best_forecasters=1,
        pre_dispatch="2*n_jobs",
        backend="loky",
        update_behaviour="full_refit",
        error_score=np.nan,
    ):
        super(ForecastingGridSearchCV, self).__init__(
            forecaster=forecaster,
            scoring=scoring,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            strategy=strategy,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            pre_dispatch=pre_dispatch,
            backend=backend,
            update_behaviour=update_behaviour,
            error_score=error_score,
        )
        self.param_grid = param_grid

    def _check_param_grid(self, param_grid):
        """_check_param_grid from sklearn 1.0.2, before it was removed."""
        if hasattr(param_grid, "items"):
            param_grid = [param_grid]

        for p in param_grid:
            for name, v in p.items():
                if isinstance(v, np.ndarray) and v.ndim > 1:
                    raise ValueError("Parameter array should be one-dimensional.")

                if isinstance(v, str) or not isinstance(v, (np.ndarray, Sequence)):
                    raise ValueError(
                        "Parameter grid for parameter ({0}) needs to"
                        " be a list or numpy array, but got ({1})."
                        " Single values need to be wrapped in a list"
                        " with one element.".format(name, type(v))
                    )

                if len(v) == 0:
                    raise ValueError(
                        "Parameter values for parameter ({0}) need "
                        "to be a non-empty sequence.".format(name)
                    )

    def _run_search(self, evaluate_candidates):
        """Search all candidates in param_grid."""
        self._check_param_grid(self.param_grid)
        return evaluate_candidates(ParameterGrid(self.param_grid))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.model_selection._split import SingleWindowSplitter
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

        params = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "param_grid": {"window_length": [2, 5]},
            "scoring": MeanAbsolutePercentageError(symmetric=True),
        }
        params2 = {
            "forecaster": PolynomialTrendForecaster(),
            "cv": SingleWindowSplitter(fh=1),
            "param_grid": {"degree": [1, 2]},
            "scoring": MeanAbsolutePercentageError(symmetric=True),
            "update_behaviour": "inner_only",
        }
        return [params, params2]


class ForecastingRandomizedSearchCV(BaseGridSearch):
    """Perform randomized-search cross-validation to find optimal model parameters.

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
    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        data ingestion strategy in fitting cv, passed to `evaluate` internally
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = forecaster is refitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update
    update_behaviour: str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update
        "full_refit" = both tuning parameters and inner estimator refit on all data seen
        "inner_only" = tuning parameters are not re-tuned, inner estimator is updated
        "no_update" = neither tuning parameters nor inner estimator are updated
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
        True = refit the forecaster with the best parameters on the entire data in fit
        False = best forecaster remains fitted on the last fold in cv
    verbose: int, optional (default=0)
    return_n_best_forecasters: int, default=1
        In case the n best forecaster should be returned, this value can be set
        and the n best forecasters will be assigned to n_best_forecasters_
    pre_dispatch: str, optional (default='2*n_jobs')
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
    pre_dispatch: str, optional (default='2*n_jobs')
    backend: str, optional (default="loky")
        Specify the parallelisation backend implementation in joblib, where
        "loky" is used by default.
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.

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
    n_best_forecasters_: list of tuples ("rank", <forecaster>)
        The "rank" is in relation to best_forecaster_
    n_best_scores_: list of float
        The scores of n_best_forecasters_ sorted from best to worst
        score of forecasters
    """

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
        return_n_best_forecasters=1,
        random_state=None,
        pre_dispatch="2*n_jobs",
        backend="loky",
        update_behaviour="full_refit",
        error_score=np.nan,
    ):
        super(ForecastingRandomizedSearchCV, self).__init__(
            forecaster=forecaster,
            scoring=scoring,
            strategy=strategy,
            n_jobs=n_jobs,
            refit=refit,
            cv=cv,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            pre_dispatch=pre_dispatch,
            backend=backend,
            update_behaviour=update_behaviour,
            error_score=error_score,
        )
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.random_state = random_state

    def _run_search(self, evaluate_candidates):
        """Search n_iter candidates from param_distributions."""
        return evaluate_candidates(
            ParameterSampler(
                self.param_distributions, self.n_iter, random_state=self.random_state
            )
        )

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict
        """
        from sktime.forecasting.model_selection._split import SingleWindowSplitter
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError

        params = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "param_distributions": {"window_length": [2, 5]},
            "scoring": MeanAbsolutePercentageError(symmetric=True),
        }

        params2 = {
            "forecaster": PolynomialTrendForecaster(),
            "cv": SingleWindowSplitter(fh=1),
            "param_distributions": {"degree": [1, 2]},
            "scoring": MeanAbsolutePercentageError(symmetric=True),
            "update_behaviour": "inner_only",
        }

        return [params, params2]
