#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements grid search functionality to tune forecasters."""

__author__ = ["mloning", "fkiraly", "aiwalter"]
__all__ = [
    "ForecastingGridSearchCV",
    "ForecastingRandomizedSearchCV",
    "ForecastingSkoptSearchCV",
]

from collections.abc import Sequence
from typing import Dict, List, Optional, Union
from warnings import warn

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import ParameterGrid, ParameterSampler, check_cv

from sktime.datatypes import mtype_to_scitype
from sktime.exceptions import NotFittedError
from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.performance_metrics.base import BaseMetric
from sktime.split.base import BaseSplitter
from sktime.utils.validation.forecasting import check_scoring


class BaseGridSearch(_DelegatedForecaster):
    _tags = {
        "scitype:y": "both",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    # todo 0.24.0: replace all tune_by_variable defaults in this file with False
    # remove deprecation message in BaseGridSearch.__init__
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
        tune_by_instance=False,
        tune_by_variable=None,
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
        self.tune_by_instance = tune_by_instance
        self.tune_by_variable = tune_by_variable

        super().__init__()

        # todo 0.24.0: remove this
        if tune_by_variable is None:
            warn(
                f"in {self.__class__.__name__}, the default for tune_by_variable "
                "will change from True to False in 0.24.0. "
                "This will tune one parameter setting for all variables, while "
                "currently it tunes one parameter per variable. "
                "In order to maintain the current behaviour, ensure to set "
                "the parameter tune_by_variable to True explicitly before upgrading "
                "to version 0.24.0.",
                DeprecationWarning,
            )
            tune_by_variable = True

        tags_to_clone = [
            "requires-fh-in-fit",
            "capability:pred_int",
            "capability:pred_int:insample",
            "capability:insample",
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

        # this ensures univariate broadcasting over variables
        # if tune_by_variable is True
        if tune_by_variable:
            self.set_tags(**{"scitype:y": "univariate"})

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "best_forecaster_"

    def _extend_to_all_scitypes(self, tagname):
        """Ensure mtypes for all scitypes are in the tag with tagname.

        Mutates self tag with name `tagname`.
        If no mtypes are present of a time series scitype, adds a pandas based one.
        If only univariate pandas scitype is present for Series ("pd.Series"),
        also adds the multivariate one ("pd.DataFrame").

        If tune_by_instance is True, only Series mtypes are added,
        and potentially present Panel or Hierarchical mtypes are removed.

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
        # if no Series mtypes are present, add pd.DataFrame
        if "Series" not in scitypes:
            tagval = tagval + ["pd.DataFrame"]
        # ensure we have a Series mtype capable of multivariate
        elif "pd.Series" in tagval and "pd.DataFrame" not in tagval:
            tagval = ["pd.DataFrame"] + tagval
        # if no Panel mtypes are present, add pd.DataFrame based one
        if "Panel" not in scitypes:
            tagval = tagval + ["pd-multiindex"]
        # if no Hierarchical mtypes are present, add pd.DataFrame based one
        if "Hierarchical" not in scitypes:
            tagval = tagval + ["pd_multiindex_hier"]

        if self.tune_by_instance:
            tagval = [x for x in tagval if mtype_to_scitype(x) == "Series"]

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
        fitted_params.update(self._get_fitted_params_default())

        return fitted_params

    def _run_search(self, evaluate_candidates):
        raise NotImplementedError("abstract method")

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
        cv = check_cv(self.cv)

        scoring = check_scoring(self.scoring, obj=self)
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
                    "Fitting {} folds for each of {} candidates,"
                    " totalling {} fits".format(
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
            self.best_forecaster_.fit(y=y, X=X, fh=fh)

        # Sort values according to rank
        results = results.sort_values(
            by=f"rank_{scoring_name}",
            ascending=True,
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
                forecaster.fit(y=y, X=X, fh=fh)
            self.n_best_forecasters_.append((rank, forecaster))
            # Save score
            score = results[f"mean_{scoring_name}"].iloc[i]
            self.n_best_scores_.append(score)

        return self

    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.Series
            Point predictions
        """
        if not self.refit:
            raise RuntimeError(
                f"In {self.__class__.__name__}, refit must be True to make predictions,"
                f" but found refit=False. If refit=False, {self.__class__.__name__} can"
                " be used only to tune hyper-parameters, as a parameter estimator."
            )
        return super()._predict(fh=fh, X=X)

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
    update_behaviour : str, optional, default = "full_refit"
        one of {"full_refit", "inner_only", "no_update"}
        behaviour of the forecaster when calling update
        "full_refit" = both tuning parameters and inner estimator refit on all data seen
        "inner_only" = tuning parameters are not re-tuned, inner estimator is updated
        "no_update" = neither tuning parameters nor inner estimator are updated
    param_grid : dict or list of dictionaries
        Model tuning parameters of the forecaster to evaluate

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        `(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float`,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    n_jobs: int, optional (default=None)
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.
    verbose: int, optional (default=0)
    return_n_best_forecasters : int, default=1
        In case the n best forecaster should be returned, this value can be set
        and the n best forecasters will be assigned to n_best_forecasters_
    pre_dispatch : str, optional (default='2*n_jobs')
    error_score : numeric value or the str 'raise', optional (default=np.nan)
        The test score returned when a forecaster fails to be fitted.
    return_train_score : bool, optional (default=False)
    backend : str, optional (default="loky")
        Specify the parallelisation backend implementation in joblib, where
        "loky" is used by default.
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.
    tune_by_instance : bool, optional (default=False)
        Whether to tune parameter by each time series instance separately,
        in case of Panel or Hierarchical data passed to the tuning estimator.
        Only applies if time series passed are Panel or Hierarchical.
        If True, clones of the forecaster will be fit to each instance separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ForecastByLevel wrapper to self.
        If False, the same best parameter is selected for all instances.
    tune_by_variable : bool, optional (default=True)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the forecaster will be fit to each variable separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ColumnEnsembleForecaster wrapper to self.
        If False, the same best parameter is selected for all variables.

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
    forecasters_ : pd.DataFramee
        DataFrame with all fitted forecasters and their parameters.
        Only present if tune_by_instance=True or tune_by_variable=True,
        and at least one of the two is applicable.
        In this case, the other attributes are not present in self,
        only in the fields of forecasters_.

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.model_selection import ForecastingGridSearchCV
    >>> from sktime.split import (
    ...     ExpandingWindowSplitter,
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
    >>> from sktime.split import ExpandingWindowSplitter
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
        tune_by_instance=False,
        tune_by_variable=None,
    ):
        super().__init__(
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
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,
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
                        "Parameter grid for parameter ({}) needs to"
                        " be a list or numpy array, but got ({})."
                        " Single values need to be wrapped in a list"
                        " with one element.".format(name, type(v))
                    )

                if len(v) == 0:
                    raise ValueError(
                        "Parameter values for parameter ({}) need "
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
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.performance_metrics.forecasting import (
            MeanAbsolutePercentageError,
            mean_absolute_percentage_error,
        )
        from sktime.split import SingleWindowSplitter

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
            "scoring": mean_absolute_percentage_error,
            "update_behaviour": "inner_only",
        }
        params3 = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "param_grid": {"window_length": [3, 4]},
            "scoring": "MeanAbsolutePercentageError(symmetric=True)",
            "update_behaviour": "no_update",
        }
        return [params, params2, params3]


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

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        `(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float`,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    n_jobs : int, optional (default=None)
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.
    verbose : int, optional (default=0)
    return_n_best_forecasters: int, default=1
        In case the n best forecaster should be returned, this value can be set
        and the n best forecasters will be assigned to n_best_forecasters_
    pre_dispatch : str, optional (default='2*n_jobs')
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
    pre_dispatch : str, optional (default='2*n_jobs')
    backend : str, optional (default="loky")
        Specify the parallelisation backend implementation in joblib, where
        "loky" is used by default.
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.
    tune_by_instance : bool, optional (default=False)
        Whether to tune parameter by each time series instance separately,
        in case of Panel or Hierarchical data passed to the tuning estimator.
        Only applies if time series passed are Panel or Hierarchical.
        If True, clones of the forecaster will be fit to each instance separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ForecastByLevel wrapper to self.
        If False, the same best parameter is selected for all instances.
    tune_by_variable : bool, optional (default=True)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the forecaster will be fit to each variable separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ColumnEnsembleForecaster wrapper to self.
        If False, the same best parameter is selected for all variables.

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
    forecasters_ : pd.DataFramee
        DataFrame with all fitted forecasters and their parameters.
        Only present if tune_by_instance=True or tune_by_variable=True,
        and at least one of the two is applicable.
        In this case, the other attributes are not present in self,
        only in the fields of forecasters_.
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
        tune_by_instance=False,
        tune_by_variable=None,
    ):
        super().__init__(
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
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,
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
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
        from sktime.split import SingleWindowSplitter

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
        params3 = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "param_distributions": {"window_length": [3, 4]},
            "scoring": "MeanAbsolutePercentageError(symmetric=True)",
            "update_behaviour": "no_update",
        }

        return [params, params2, params3]


class ForecastingSkoptSearchCV(BaseGridSearch):
    """Bayesian search over hyperparameters for a forecaster.

    Experimental: This feature is under development and interface may likely to change.

    Parameters
    ----------
    forecaster : estimator object.
        The estimator should implement the sktime or scikit-learn estimator interface.
        Either the estimator must contain a "score" function,
        or a scoring function must be passed.
    cv : cross-validation generator or an iterable
        Splitter used for generating validation folds.
        e.g. SlidingWindowSplitter()
    param_distributions : dict or a list of dict/tuple. See below for details.
        1. If dict, a dictionary that represents the search space over the parameters of
        the provided estimator. The keys are parameter names (strings), and the values
        follows the following format. A list to store categorical parameters and a
        tuple for integer and real parameters with the following format
        (int/float, int/float, "prior") e.g (1e-6, 1e-1, "log-uniform").
        2. If a list of dict, each dictionary corresponds to a parameter space,
        following the same structure described in case 1 above. the search will be
        performed sequentially for each parameter space, with the number of samples
        set to n_iter.
        3. If a list of tuple, tuple must contain (dict, int) where the int refers to
        n_iter for that search space. dict must follow the same structure as in case 1.
        This is useful if you want to perform a search with different number of
        iterations for each parameter space.
    n_iter : int, default=10
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution. Consider increasing n_points
        if you want to try more parameter settings in parallel.
    n_points : int, default=1
        Number of parameter settings to sample in parallel.
        If this does not align with n_iter, the last iteration will sample less points

    scoring : sktime metric (BaseMetric), str, or callable, optional (default=None)
        scoring metric to use in tuning the forecaster

        * sktime metric objects (BaseMetric) descendants can be searched
        with the ``registry.all_estimators`` search utility,
        for instance via ``all_estimators("metric", as_dataframe=True)``

        * If callable, must have signature
        `(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float`,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    optimizer_kwargs: dict, optional
        Arguments passed to Optimizer to control the bahaviour of the bayesian search.
        For example, {'base_estimator': 'RF'} would use a Random Forest surrogate
        instead of the default Gaussian Process. Please refer to the `skopt.Optimizer`
        documentation for more information.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
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
    refit : bool, optional (default=True)
        True = refit the forecaster with the best parameters on the entire data in fit
        False = no refitting takes place. The forecaster cannot be used to predict.
        This is to be used to tune the hyperparameters, and then use the estimator
        as a parameter estimator, e.g., via get_fitted_params or PluginParamsForecaster.
    verbose : int, optional (default=0)
    error_score : "raise" or numeric, default=np.nan
        Value to assign to the score if an exception occurs in estimator fitting. If set
        to "raise", the exception is raised. If a numeric value is given,
        FitFailedWarning is raised.
    return_n_best_forecasters: int, default=1
        In case the n best forecaster should be returned, this value can be set
        and the n best forecasters will be assigned to n_best_forecasters_
    pre_dispatch : str, optional (default='2*n_jobs')
    n_jobs : int, optional (default=None)
        Number of jobs to run in parallel.
        None means 1 unless in a joblib.parallel_backend context.
        -1 means using all processors.
    backend : str, optional (default="loky")
        Specify the parallelisation backend implementation in joblib, where
        "loky" is used by default.
    tune_by_instance : bool, optional (default=False)
        Whether to tune parameter by each time series instance separately,
        in case of Panel or Hierarchical data passed to the tuning estimator.
        Only applies if time series passed are Panel or Hierarchical.
        If True, clones of the forecaster will be fit to each instance separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ForecastByLevel wrapper to self.
        If False, the same best parameter is selected for all instances.
    tune_by_variable : bool, optional (default=True)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the forecaster will be fit to each variable separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ColumnEnsembleForecaster wrapper to self.
        If False, the same best parameter is selected for all variables.

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
    forecasters_ : pd.DataFramee
        DataFrame with all fitted forecasters and their parameters.
        Only present if tune_by_instance=True or tune_by_variable=True,
        and at least one of the two is applicable.
        In this case, the other attributes are not present in self,
        only in the fields of forecasters_.

    Examples
    --------
    >>> from sktime.datasets import load_shampoo_sales
    >>> from sktime.forecasting.model_selection import ForecastingSkoptSearchCV
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sklearn.ensemble import GradientBoostingRegressor
    >>> from sktime.forecasting.compose import make_reduction
    >>> y = load_shampoo_sales()
    >>> fh = [1,2,3,4]
    >>> cv = ExpandingWindowSplitter(fh=fh)
    >>> forecaster = make_reduction(GradientBoostingRegressor(random_state=10))
    >>> param_distributions = {
    ...     "estimator__learning_rate" : (1e-4, 1e-1, "log-uniform"),
    ...     "window_length" : (1, 10, "uniform"),
    ...     "estimator__criterion" : ["friedman_mse", "squared_error"]}
    >>> sscv = ForecastingSkoptSearchCV(
    ...     forecaster=forecaster,
    ...     param_distributions=param_distributions,
    ...     cv=cv,
    ...     n_iter=5,
    ...     random_state=10)  # doctest: +SKIP
    >>> sscv.fit(y)  # doctest: +SKIP
    ForecastingSkoptSearchCV(...)
    >>> y_pred = sscv.predict(fh)  # doctest: +SKIP
    """

    _tags = {
        "scitype:y": "both",
        "requires-fh-in-fit": False,
        "handles-missing-data": False,
        "ignores-exogeneous-X": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["numpy<1.24", "scikit-optimize"],
        "python_version": ">= 3.6",
        "python_dependencies_alias": {"scikit-optimize": "skopt"},
    }

    def __init__(
        self,
        forecaster,
        cv: BaseSplitter,
        param_distributions: Union[Dict, List[Dict]],
        n_iter: int = 10,
        n_points: Optional[int] = 1,
        random_state: Optional[int] = None,
        scoring: Optional[List[BaseMetric]] = None,
        optimizer_kwargs: Optional[Dict] = None,
        strategy: Optional[str] = "refit",
        n_jobs: Optional[int] = None,
        refit: bool = True,
        verbose: int = 0,
        return_n_best_forecasters: int = 1,
        pre_dispatch: str = "2*n_jobs",
        backend: str = "loky",
        update_behaviour: str = "full_refit",
        error_score=np.nan,
        tune_by_instance=False,
        tune_by_variable=None,
    ):
        self.param_distributions = param_distributions
        self.n_iter = n_iter
        self.n_points = n_points
        self.random_state = random_state
        self.optimizer_kwargs = optimizer_kwargs
        super().__init__(
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
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,
        )

    def _fit(self, y, X=None, fh=None):
        """Run fit with all sets of parameters."""
        self._check_cv = check_cv(self.cv)
        self._check_scoring = check_scoring(self.scoring, obj=self)
        scoring_name = f"test_{self._check_scoring.name}"
        self._check_search_space(self.param_distributions)
        self.cv_results_ = pd.DataFrame()

        self._run_search(y, X)

        # Rank results, according to whether greater is better for the given scoring.
        self.cv_results_[f"rank_{scoring_name}"] = self.cv_results_.loc[
            :, f"mean_{scoring_name}"
        ].rank(ascending=self._check_scoring.get_tag("lower_is_better"))

        results = self.cv_results_
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
            by=f"rank_{scoring_name}",
            ascending=True,
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

    def _run_search(self, y, X=None, fh=None):
        """Search n_iter candidates from param_distributions.

        Parameters
        ----------
        y : time series in sktime compatible data container format
            Target time series to which to fit the forecaster.
        X : time series in sktime compatible format, optional (default=None)
            Exogenous variables.
        fh : int, list, np.array or ForecastingHorizon, optional (default=None)
        """
        # check if space is a single dict, convert to list if so
        param_distributions = self.param_distributions
        if isinstance(param_distributions, dict):
            param_distributions = [param_distributions]

        if self.optimizer_kwargs is None:
            self.optimizer_kwargs_ = {}
        else:
            self.optimizer_kwargs_ = dict(self.optimizer_kwargs)
        self.optimizer_kwargs_["random_state"] = self.random_state

        optimizers = []
        mappings = []
        for search_space in param_distributions:
            if isinstance(search_space, tuple):
                search_space = search_space[0]

            # hacky approach to handle unhashable type objects
            if "forecaster" in search_space:
                forecasters = search_space.get("forecaster")
                mapping = {
                    num: estimator for num, estimator in enumerate(forecasters)
                }  # noqa
                search_space["forecaster"] = list(mapping.keys())
                mappings.append(mapping)
            else:
                mappings.append(None)

            optimizers.append(self._create_optimizer(search_space))
        self.optimizers_ = optimizers  # will save the states of the optimizers

        if self.verbose > 0:
            n_candidates = self.n_iter
            n_splits = self.cv.get_n_splits(y)
            print(  # noqa
                "Fitting {} folds for each of {} candidates,"
                " totalling {} fits".format(
                    n_splits, n_candidates, n_candidates * n_splits
                )
            )

        # Run sequential search by iterating through each optimizer and evaluates
        # the search space iteratively until all n_iter candidates are evaluated.
        for num, (search_space, optimizer) in enumerate(
            zip(param_distributions, optimizers)
        ):
            # if search subspace n_iter is provided, use it otherwise use self.n_iter
            if isinstance(search_space, tuple):
                n_iter = search_space[1]
            else:
                n_iter = self.n_iter

            # iterations for each search space
            while n_iter > 0:
                # when n_iter < n_points points left for evaluation
                n_points_adjusted = min(n_iter, self.n_points)
                self._evaluate_step(
                    y,
                    X,
                    optimizer,
                    n_points=n_points_adjusted,
                    mapping=mappings[num],
                )
                n_iter -= self.n_points
            # reset n_iter for next search space
            n_iter = self.n_iter

    def _evaluate_step(self, y, X, optimizer, n_points, mapping=None):
        """Evaluate a candidate parameter set at each iteration.

        Parameters
        ----------
        y : time series in sktime compatible data container format
            Target time series to which to fit the forecaster.
        X : time series in sktime compatible format, optional (default=None)
            Exogenous variables.
        optimizer : skopt.Optimizer
            Optimizer instance.
        n_points : int
            Number of candidate parameter combination to evaluate at each step.
            if n_points=2, then the two candidate parameter combinations are evaluated
            e.g {'sp': 1, 'strategy':'last'} and {'sp': 2, 'strategy': 'mean'}.
        mapping : dict, optional (default=None)
            Mapping of forecaster to estimator instance.
        """
        from skopt.utils import use_named_args

        # Get a list of dimension parameter space with name from optimizer
        dimensions = optimizer.space.dimensions
        test_score_name = f"test_{self._check_scoring.name}"

        @use_named_args(dimensions)  # decorater to convert candidate param list to dict
        def _fit_and_score(**params):
            # Clone forecaster.
            forecaster = self.forecaster.clone()

            # map forecaster back to estimator instance
            if "forecaster" in params:
                params["forecaster"] = mapping[params["forecaster"]]

            # Set parameters.
            forecaster.set_params(**params)

            # Evaluate.
            out = evaluate(
                forecaster=forecaster,
                cv=self._check_cv,
                y=y,
                X=X,
                strategy=self.strategy,
                scoring=self._check_scoring,
                error_score=self.error_score,
            )

            # Filter columns.
            out = out.filter(
                items=[test_score_name, "fit_time", "pred_time"],
                axis=1,
            )

            # Aggregate results.
            out = out.mean()
            out = out.add_prefix("mean_")

            # Add parameters to output table.
            out["params"] = params

            return out

        parallel = Parallel(
            n_jobs=self.n_jobs, pre_dispatch=self.pre_dispatch, backend=self.backend
        )

        candidate_params = optimizer.ask(n_points=n_points)
        out = parallel(delayed(_fit_and_score)(params) for params in candidate_params)

        # fetch the mean evaluation metrics and feed them back to optimizer
        results_df = pd.DataFrame(out)
        # as the optimizer is minimising a score,
        # we need to negate the score if higher_is_better
        mean_test_score = results_df["mean_" + test_score_name]
        if self._check_scoring.get_tag("lower_is_better"):
            scores = list(mean_test_score)
        else:
            scores = list(-mean_test_score)
        # Update optimizer with evaluation metrics.
        optimizer.tell(candidate_params, scores)
        # keep updating the cv_results_ attribute by concatinating the result dataframe
        self.cv_results_ = pd.concat([self.cv_results_, results_df], ignore_index=True)

        try:
            assert len(out) >= 1
        except AssertionError:
            raise ValueError(
                "No fits were performed. "
                "Was the CV iterator empty? "
                "Were there no candidates?"
            )

    def _create_optimizer(self, params_space):
        """Instantiate optimizer for a search parameter space.

        Responsible for initialising optimizer with the correct parameters
        names and values.

        Parameters
        ----------
        params_space : dict
            Dictionary with parameters names (string) as keys and the values are
            instances of skopt.space.Dimension (Real, Integer, or Categorical)

        Returns
        -------
        optimizer : skopt.Optimizer
        """
        from skopt.optimizer import Optimizer
        from skopt.utils import dimensions_aslist

        kwargs = self.optimizer_kwargs_.copy()
        # convert params space to a list ordered by the key name
        kwargs["dimensions"] = dimensions_aslist(params_space)
        dimensions_name = list(sorted(params_space.keys()))
        optimizer = Optimizer(**kwargs)
        # set the name of the dimensions if not set
        for i in range(len(optimizer.space.dimensions)):
            if optimizer.space.dimensions[i].name is not None:
                continue
            optimizer.space.dimensions[i].name = dimensions_name[i]

        return optimizer

    def _check_search_space(self, search_space):
        """Check whether the search space argument is correct.

        from skopt.BayesSearchCV._check_search_space
        """
        from skopt.space import check_dimension

        if len(search_space) == 0:
            raise ValueError(
                "The search_spaces parameter should contain at least one"
                "non-empty search space, got %s" % search_space
            )

        # check if space is a single dict, convert to list if so
        if isinstance(search_space, dict):
            search_space = [search_space]

        # check if the structure of the space is proper
        if isinstance(search_space, list):
            # convert to just a list of dicts
            dicts_only = []

            # 1. check the case when a tuple of space, n_iter is provided
            for elem in search_space:
                if isinstance(elem, tuple):
                    if len(elem) != 2:
                        raise ValueError(
                            "All tuples in list of search spaces should have"
                            "length 2, and contain (dict, int), got %s" % elem
                        )
                    subspace, n_iter = elem

                    if (not isinstance(n_iter, int)) or n_iter < 0:
                        raise ValueError(
                            "Number of iterations in search space should be"
                            "positive integer, got %s in tuple %s " % (n_iter, elem)
                        )

                    # save subspaces here for further checking
                    dicts_only.append(subspace)
                elif isinstance(elem, dict):
                    dicts_only.append(elem)
                else:
                    raise TypeError(
                        "A search space should be provided as a dict or"
                        "tuple (dict, int), got %s" % elem
                    )

            # 2. check all the dicts for correctness of contents
            for subspace in dicts_only:
                for params_name, param_value in subspace.items():
                    if params_name != "forecaster":
                        check_dimension(param_value)
        else:
            raise TypeError(
                "Search space should be provided as a dict or list of dict,"
                "got %s" % search_space
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
        from sktime.forecasting.naive import NaiveForecaster
        from sktime.forecasting.trend import PolynomialTrendForecaster
        from sktime.performance_metrics.forecasting import MeanAbsolutePercentageError
        from sktime.split import SingleWindowSplitter

        params = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "param_distributions": {"window_length": [2, 5]},
            "scoring": MeanAbsolutePercentageError(symmetric=True),
            "n_iter": 1,
        }

        params2 = {
            "forecaster": PolynomialTrendForecaster(),
            "cv": SingleWindowSplitter(fh=1),
            "param_distributions": {"degree": [1, 2]},
            "scoring": MeanAbsolutePercentageError(symmetric=True),
            "update_behaviour": "inner_only",
            "n_iter": 1,
        }

        return [params, params2]
