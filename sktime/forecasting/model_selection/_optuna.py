# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Parameter search via optuna."""

from collections.abc import Iterable, Mapping

import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv

from sktime.exceptions import NotFittedError
from sktime.forecasting.model_selection._base import BaseGridSearch, _fit_and_score
from sktime.utils.validation.forecasting import check_scoring
from sktime.utils.warnings import warn


class ForecastingOptunaSearchCV(BaseGridSearch):
    """Perform Optuna search cross-validation to find optimal model hyperparameters.

    Experimental: This feature is under development and interfaces may change.

    In ``fit``, this estimator uses the ``optuna`` base search algorithm
    applied to the ``sktime`` ``evaluate`` benchmarking output.

    ``param_grid`` is used to parametrize the search space, over parameters of
    the passed ``forecaster``, via ``set_params``.

    The remaining parameters are passed directly to ``evaluate``, to obtain
    the primary optimization outcome as the aggregate ``scoring`` metric specified
    on the evaluation schema.

    Parameters
    ----------
    forecaster : sktime forecaster, BaseForecaster instance or interface compatible
        The forecaster to tune, must implement the sktime forecaster interface.
        sklearn regressors can be used, but must first be converted to forecasters
        via one of the reduction compositors, e.g., via ``make_reduction``
    cv : cross-validation generator or an iterable
        Splitter used for generating validation folds.
        e.g. ExpandingWindowSplitter()
    param_grid : dict of optuna samplers
        Dictionary with parameters names as keys and lists of parameter distributions
        from which to sample parameter values.
        e.g. {"forecaster": optuna.distributions.CategoricalDistribution(
        (STLForecaster(), ThetaForecaster())}
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

    strategy : {"refit", "update", "no-update_params"}, optional, default="refit"
        data ingestion strategy in fitting cv, passed to ``evaluate`` internally
        defines the ingestion mode when the forecaster sees new data when window expands
        "refit" = a new copy of the forecaster is fitted to each training window
        "update" = forecaster is updated with training window data, in sequence provided
        "no-update_params" = fit to first training window, re-used without fit or update
    refit : bool, default=True
        Refit an estimator using the best found parameters on the whole dataset.
    verbose : int, default=0
        Controls the verbosity: the higher, the more messages.
    return_n_best_forecasters : int, default=1
        Number of best forecasters to return.
    backend : str, default="loky"
        Backend to use when running the fit.
    update_behaviour : str, default="full_refit"
        Determines how to update the forecaster during fitting.
    error_score : 'raise' or numeric, default=np.nan
        Value to assign to the score if an error occurs in estimator fitting.
    n_evals : int, default=100
        Number of parameter settings that are sampled. n_iter trades
        off runtime vs quality of the solution.
    sampler : Optuna sampler, optional (default=None)
        e.g. optuna.samplers.TPESampler(seed=42)

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

    Examples
    --------
    >>> from sktime.forecasting.model_selection import (
    ...     ForecastingOptunaSearchCV,
    ...     )
    >>> from sktime.datasets import load_shampoo_sales
    >>> import warnings
    >>> warnings.simplefilter(action="ignore", category=FutureWarning)
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.split import ExpandingWindowSplitter
    >>> from sktime.split import temporal_train_test_split
    >>> from sklearn.preprocessing import MinMaxScaler, RobustScaler
    >>> from sktime.forecasting.compose import TransformedTargetForecaster
    >>> from sktime.transformations.series.adapt import TabularToSeriesAdaptor
    >>> from sktime.transformations.series.detrend import Deseasonalizer, Detrender
    >>> from sktime.forecasting.naive import NaiveForecaster
    >>> from sktime.forecasting.trend import STLForecaster, TrendForecaster
    >>> import optuna
    >>> from  optuna.distributions import CategoricalDistribution

    >>> y = load_shampoo_sales()
    >>> y_train, y_test = temporal_train_test_split(y=y, test_size=6)
    >>> fh = ForecastingHorizon(y_test.index, is_relative=False).to_relative(
    ...         cutoff=y_train.index[-1]
    ...     )
    >>> cv = ExpandingWindowSplitter(fh=fh, initial_window=24, step_length=1)
    >>> forecaster = TransformedTargetForecaster(
    ...     steps=[
    ...             ("detrender", Detrender()),
    ...             ("scaler", RobustScaler()),
    ...             ("minmax2", MinMaxScaler((1, 10))),
    ...             ("forecaster", NaiveForecaster()),
    ...         ]
    ...     )
    >>> param_grid = {
    ...     "scaler__with_scaling": CategoricalDistribution(
    ...             (True, False)
    ...         ),
    ...     "forecaster": CategoricalDistribution(
    ...             (NaiveForecaster(), TrendForecaster())
    ...         ),
    ...     }
    >>> gscv = ForecastingOptunaSearchCV(
    ...         forecaster=forecaster,
    ...         param_grid=param_grid,
    ...         cv=cv,
    ...         n_evals=10,
    ...     )
    >>> gscv.fit(y)
    ForecastingOptunaSearchCV(...)
    >>> print(f"{gscv.best_params_=}")  # doctest: +SKIP
    """

    _tags = {
        "authors": ["gareth-brown-86", "mk406", "bastisar"],
        "maintainers": ["gareth-brown-86", "mk406"],
        "scitype:y": "both",
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:exogenous": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "python_dependencies": ["optuna"],
        "python_version": ">= 3.6",
        # CI and test flags
        # -----------------
        "tests:libs": ["sktime.forecasting.model_selection._base"],
    }

    def __init__(
        self,
        forecaster,
        cv,
        param_grid,
        scoring=None,
        strategy="refit",
        refit=True,
        verbose=0,
        return_n_best_forecasters=1,
        backend="loky",
        update_behaviour="full_refit",
        error_score=np.nan,
        n_evals=100,
        sampler=None,
    ):
        super().__init__(
            forecaster=forecaster,
            scoring=scoring,
            refit=refit,
            cv=cv,
            strategy=strategy,
            verbose=verbose,
            return_n_best_forecasters=return_n_best_forecasters,
            backend=backend,
            update_behaviour=update_behaviour,
            error_score=error_score,
        )
        self.param_grid = param_grid
        self.n_evals = n_evals
        self.sampler = sampler

        warn(
            "ForecastingOptunaSearchCV is experimental, and interfaces may change. "
            "User feedback and suggestions for future development "
            "are appreciated in issue #6618 here: "
            "https://github.com/sktime/sktime/issues/6618",
            obj=self,
            stacklevel=2,
        )

    def _fit(self, y, X=None, fh=None):
        cv = check_cv(self.cv)
        scoring = check_scoring(self.scoring, obj=self)
        scoring_name = f"test_{scoring.name}"
        sampler = self.sampler

        if not isinstance(self.param_grid, (Mapping, Iterable)):
            raise TypeError(
                "Parameter distribution is not a dict or a list,"
                f" got: {self.param_grid!r} of type "
                f"{type(self.param_grid).__name__}"
            )

        if isinstance(self.param_grid, Mapping):
            # wrap dictionary in a singleton list to support either dict
            # or list of dicts
            self._param_grid = [self.param_grid]
        else:
            self._param_grid = self.param_grid

        results = self._run_search(
            y,
            X,
            cv,
            scoring,
            scoring_name,
            sampler,
        )

        results[f"rank_{scoring_name}"] = results[f"mean_{scoring_name}"].rank(
            ascending=scoring.get_tag("lower_is_better")
        )
        self.cv_results_ = results
        self.best_index_ = results.loc[:, f"rank_{scoring_name}"].argmin()
        if self.best_index_ == -1:
            raise NotFittedError(
                f"""All fits of forecaster failed,
                set error_score='raise' to see the exceptions.
                Failed forecaster: {self.forecaster}"""
            )
        self.best_score_ = results.loc[self.best_index_, f"mean_{scoring_name}"]
        self.best_params_ = results.loc[self.best_index_, "params"]
        self.best_forecaster_ = self.forecaster.clone().set_params(**self.best_params_)

        if self.refit:
            self.best_forecaster_.fit(y, X, fh)

        results = results.sort_values(
            by=f"mean_{scoring_name}", ascending=scoring.get_tag("lower_is_better")
        )
        self.n_best_forecasters_ = []
        self.n_best_scores_ = []
        for i in range(self.return_n_best_forecasters):
            params = results["params"].iloc[i]
            rank = results[f"rank_{scoring_name}"].iloc[i]
            rank = str(int(rank))
            forecaster = self.forecaster.clone().set_params(**params)
            if self.refit:
                forecaster.fit(y, X, fh)
            self.n_best_forecasters_.append((rank, forecaster))
            score = results[f"mean_{scoring_name}"].iloc[i]
            self.n_best_scores_.append(score)

        return self

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
        from sktime.utils.dependencies import _check_soft_dependencies

        if not _check_soft_dependencies("optuna", severity="none"):
            return {
                "forecaster": "foo",
                "cv": "bar",
                "param_grid": "foobar",
                "scoring": "barfoo",
            }

        from optuna.distributions import CategoricalDistribution

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
            "param_grid": {"window_length": CategoricalDistribution((2, 5))},
            "scoring": MeanAbsolutePercentageError(symmetric=True),
        }
        params2 = {
            "forecaster": PolynomialTrendForecaster(),
            "cv": SingleWindowSplitter(fh=1),
            "param_grid": {"degree": CategoricalDistribution((1, 2))},
            "scoring": mean_absolute_percentage_error,
            "update_behaviour": "inner_only",
        }
        params3 = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "param_grid": {"window_length": CategoricalDistribution((3, 4))},
            "scoring": "MeanAbsolutePercentageError(symmetric=True)",
            "update_behaviour": "no_update",
        }
        scorer_with_lower_is_better_false = MeanAbsolutePercentageError(symmetric=True)
        scorer_with_lower_is_better_false.set_tags(**{"lower_is_better": False})
        params4 = {
            "forecaster": NaiveForecaster(strategy="mean"),
            "cv": SingleWindowSplitter(fh=1),
            "param_grid": {"window_length": CategoricalDistribution((2, 5))},
            "scoring": scorer_with_lower_is_better_false,
        }
        return [params, params2, params3, params4]

    def _get_score(self, out, scoring_name):
        return out[f"mean_{scoring_name}"]

    def _run_search(self, y, X, cv, scoring, scoring_name, sampler):
        import optuna

        all_results = []  # List to store results from all parameter grids

        for (
            param_grid_dict
        ) in self._param_grid:  # Assuming self._param_grid is now a list of dicts
            scoring_direction = (
                "minimize" if scoring.get_tag("lower_is_better") else "maximize"
            )
            study = optuna.create_study(direction=scoring_direction, sampler=sampler)
            meta = {}
            meta["forecaster"] = self.forecaster
            meta["y"] = y
            meta["X"] = X
            meta["cv"] = cv
            meta["strategy"] = self.strategy
            meta["scoring"] = scoring
            meta["error_score"] = self.error_score
            meta["scoring_name"] = scoring_name
            for _ in range(self.n_evals):
                trial = study.ask(param_grid_dict)
                params = {
                    name: trial.params[name] for name, v in param_grid_dict.items()
                }

                out = _fit_and_score(params, meta)

                study.tell(trial, self._get_score(out, scoring_name))

            params_list = [trial.params for trial in study.trials]

            results = study.trials_dataframe()

            # Add the parameters as a new column to the DataFrame
            results["params"] = params_list
            all_results.append(results)  # Append the results DataFrame to the list

        # Combine all results into a single DataFrame
        combined_results = pd.concat(all_results, ignore_index=True)
        return combined_results.rename(columns={"value": f"mean_{scoring_name}"})
