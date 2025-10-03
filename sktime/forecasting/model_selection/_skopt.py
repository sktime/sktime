# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter search via scikit-optimize."""

import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv

from sktime.exceptions import NotFittedError
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection._base import BaseGridSearch
from sktime.performance_metrics.base import BaseMetric
from sktime.split.base import BaseSplitter
from sktime.utils.parallel import parallelize
from sktime.utils.validation.forecasting import check_scoring


class ForecastingSkoptSearchCV(BaseGridSearch):
    """Bayesian search over hyperparameters for a forecaster.

    Experimental: This feature is under development and interface may likely to change.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.
        sklearn regressors can be used, but must first be converted to forecasters
        via one of the reduction compositors, e.g., via ``make_reduction``
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
        ``(y_true: 1D np.ndarray, y_pred: 1D np.ndarray) -> float``,
        assuming np.ndarrays being of the same length, and lower being better.
        Metrics in sktime.performance_metrics.forecasting are all of this form.

        * If str, uses registry.resolve_alias to resolve to one of the above.
          Valid strings are valid registry.craft specs, which include
          string repr-s of any BaseMetric object, e.g., "MeanSquaredError()";
          and keys of registry.ALIAS_DICT referring to metrics.

        * If None, defaults to MeanAbsolutePercentageError()

    optimizer_kwargs: dict, optional
        Arguments passed to Optimizer to control the behaviour of the bayesian search.
        For example, {'base_estimator': 'RF'} would use a Random Forest surrogate
        instead of the default Gaussian Process. Please refer to the ``skopt.Optimizer``
        documentation for more information.
    random_state : int, RandomState instance or None, default=None
        Pseudo random number generator state used for random uniform sampling
        from lists of possible values instead of scipy.stats distributions.
        Pass an int for reproducible output across multiple
        function calls.
    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        data ingestion strategy in fitting cv, passed to ``evaluate`` internally
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = a new copy of the forecaster is fitted to each training window
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
        and the n best forecasters will be assigned to n_best_forecasters_.
        Set return_n_best_forecasters to -1 to return all forecasters.

    backend : {"dask", "loky", "multiprocessing", "threading"}, by default "loky".
        Runs parallel evaluate if specified and ``strategy`` is set as "refit".

        - "None": executes loop sequentally, simple list comprehension
        - "loky", "multiprocessing" and "threading": uses ``joblib.Parallel`` loops
        - "joblib": custom and 3rd party ``joblib`` backends, e.g., ``spark``
        - "dask": uses ``dask``, requires ``dask`` package in environment
        - "ray": uses ``ray``, requires ``ray`` package in environment

        Recommendation: Use "dask" or "loky" for parallel evaluate.
        "threading" is unlikely to see speed ups due to the GIL and the serialization
        backend (``cloudpickle``) for "dask" and "loky" is generally more robust
        than the standard ``pickle`` library used in "multiprocessing".

    tune_by_instance : bool, optional (default=False)
        Whether to tune parameter by each time series instance separately,
        in case of Panel or Hierarchical data passed to the tuning estimator.
        Only applies if time series passed are Panel or Hierarchical.
        If True, clones of the forecaster will be fit to each instance separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ForecastByLevel wrapper to self.
        If False, the same best parameter is selected for all instances.
    tune_by_variable : bool, optional (default=False)
        Whether to tune parameter by each time series variable separately,
        in case of multivariate data passed to the tuning estimator.
        Only applies if time series passed are strictly multivariate.
        If True, clones of the forecaster will be fit to each variable separately,
        and are available in fields of the forecasters_ attribute.
        Has the same effect as applying ColumnEnsembleForecaster wrapper to self.
        If False, the same best parameter is selected for all variables.

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
        - "dask": any valid keys for ``dask.compute`` can be passed, e.g., ``scheduler``

        - "ray": The following keys can be passed:

            - "ray_remote_args": dictionary of valid keys for ``ray.init``
            - "shutdown_ray": bool, default=True; False prevents ``ray`` from shutting
                down after parallelization.
            - "logger_name": str, default="ray"; name of the logger to use.
            - "mute_warnings": bool, default=False; if True, suppresses warnings

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
        # packaging info
        # --------------
        "authors": ["HazrulAkmal"],
        "maintainers": ["HazrulAkmal"],
        "python_dependencies": ["scikit-optimize"],
        "python_version": ">= 3.6",
        # estimator type
        # --------------
        "scitype:y": "both",
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:exogenous": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "capability:random_state": True,
        "property:randomness": "derandomized",
        # CI and test flags
        # -----------------
        "tests:vm": True,  # run on separate VM since scikit-optimize is deprecated
        "tests:libs": ["sktime.forecasting.model_selection._base"],
    }

    def __init__(
        self,
        forecaster,
        cv: BaseSplitter,
        param_distributions: dict | list[dict],
        n_iter: int = 10,
        n_points: int | None = 1,
        random_state: int | None = None,
        scoring: list[BaseMetric] | None = None,
        optimizer_kwargs: dict | None = None,
        strategy: str | None = "refit",
        refit: bool = True,
        verbose: int = 0,
        return_n_best_forecasters: int = 1,
        backend: str = "loky",
        update_behaviour: str = "full_refit",
        error_score=np.nan,
        tune_by_instance=False,
        tune_by_variable=False,
        backend_params=None,
        n_jobs="deprecated",
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
            refit=refit,
            cv=cv,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            backend=backend,
            update_behaviour=update_behaviour,
            error_score=error_score,
            tune_by_instance=tune_by_instance,
            tune_by_variable=tune_by_variable,
            backend_params=backend_params,
            n_jobs=n_jobs,
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
                mapping = {num: estimator for num, estimator in enumerate(forecasters)}
                search_space["forecaster"] = list(mapping.keys())
                mappings.append(mapping)
            else:
                mappings.append(None)

            optimizers.append(self._create_optimizer(search_space))
        self.optimizers_ = optimizers  # will save the states of the optimizers

        if self.verbose > 0:
            n_candidates = self.n_iter
            n_splits = self.cv.get_n_splits(y)
            print(
                f"Fitting {n_splits} folds for each of {n_candidates} candidates,"
                f" totalling {n_candidates * n_splits} fits"
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
        # Get a list of dimension parameter space with name from optimizer
        dimensions = optimizer.space.dimensions
        test_score_name = f"test_{self._check_scoring.name}"

        # Set meta variables for parallelization.
        meta = {}
        meta["forecaster"] = self.forecaster
        meta["y"] = y
        meta["X"] = X
        meta["mapping"] = mapping
        meta["cv"] = self._check_cv
        meta["strategy"] = self.strategy
        meta["scoring"] = self._check_scoring
        meta["error_score"] = self.error_score
        meta["test_score_name"] = test_score_name
        meta["dimensions"] = dimensions

        candidate_params = optimizer.ask(n_points=n_points)

        out = parallelize(
            fun=_fit_and_score_skopt,
            iter=candidate_params,
            meta=meta,
            backend=self.backend,
            backend_params=self.backend_params,
        )

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
        # keep updating the cv_results_ attribute by concatenating the result dataframe
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
        dimensions_name = sorted(params_space.keys())
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
            special parameters are defined for a value, will return ``"default"`` set.

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


def _fit_and_score_skopt(params, meta):
    from skopt.utils import use_named_args

    y = meta["y"]
    X = meta["X"]
    cv = meta["cv"]
    mapping = meta["mapping"]
    strategy = meta["strategy"]
    scoring = meta["scoring"]
    error_score = meta["error_score"]
    dimensions = meta["dimensions"]
    test_score_name = meta["test_score_name"]

    @use_named_args(dimensions)  # decorator to convert candidate param list to dict
    def _fit_and_score(**params):
        # Clone forecaster.
        forecaster = meta["forecaster"].clone()

        # map forecaster back to estimator instance
        if "forecaster" in params:
            params["forecaster"] = mapping[params["forecaster"]]

        # Set parameters.
        forecaster.set_params(**params)

        # Evaluate.
        out = evaluate(
            forecaster=forecaster,
            cv=cv,
            y=y,
            X=X,
            strategy=strategy,
            scoring=scoring,
            error_score=error_score,
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

    return _fit_and_score(params)
