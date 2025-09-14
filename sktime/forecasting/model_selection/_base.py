"""Base class for forecasting tuners."""

import numpy as np
import pandas as pd
from sklearn.model_selection import check_cv

from sktime.datatypes import mtype_to_scitype
from sktime.exceptions import NotFittedError
from sktime.forecasting.base._delegate import _DelegatedForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.utils.parallel import parallelize
from sktime.utils.validation.forecasting import check_scoring
from sktime.utils.warnings import warn


class BaseGridSearch(_DelegatedForecaster):
    _tags = {
        "authors": ["mloning", "fkiraly", "aiwalter"],
        "scitype:y": "both",
        "requires-fh-in-fit": False,
        "capability:missing_values": False,
        "capability:exogenous": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
    }

    def __init__(
        self,
        forecaster,
        cv,
        strategy="refit",
        backend="loky",
        refit=False,
        scoring=None,
        verbose=0,
        return_n_best_forecasters=1,
        update_behaviour="full_refit",
        error_score=np.nan,
        tune_by_instance=False,
        tune_by_variable=False,
        backend_params=None,
        n_jobs="deprecated",
    ):
        self.forecaster = forecaster
        self.cv = cv
        self.strategy = strategy
        self.backend = backend
        self.refit = refit
        self.scoring = scoring
        self.verbose = verbose
        self.return_n_best_forecasters = return_n_best_forecasters
        self.update_behaviour = update_behaviour
        self.error_score = error_score
        self.tune_by_instance = tune_by_instance
        self.tune_by_variable = tune_by_variable
        self.backend_params = backend_params
        self.n_jobs = n_jobs

        super().__init__()

        self._set_delegated_tags(forecaster)

        tags_to_clone = ["y_inner_mtype", "X_inner_mtype"]
        self.clone_tags(forecaster, tags_to_clone)
        self._extend_to_all_scitypes("y_inner_mtype")
        self._extend_to_all_scitypes("X_inner_mtype")

        # this ensures univariate broadcasting over variables
        # if tune_by_variable is True
        if tune_by_variable:
            self.set_tags(**{"scitype:y": "univariate"})

        # todo 1.0.0: check if this is still necessary
        # n_jobs is deprecated, left due to use in tutorials, books, blog posts
        if n_jobs != "deprecated":
            warn(
                f"Parameter n_jobs of {self.__class__.__name__} has been removed "
                "in sktime 0.27.0 and is no longer used. It is ignored when passed. "
                "Instead, the backend and backend_params parameters should be used "
                "to pass n_jobs or other parallelization parameters.",
                obj=self,
                stacklevel=2,
            )

    # attribute for _DelegatedForecaster, which then delegates
    #     all non-overridden methods are same as of getattr(self, _delegate_name)
    #     see further details in _DelegatedForecaster docstring
    _delegate_name = "best_forecaster_"

    def _extend_to_all_scitypes(self, tagname):
        """Ensure mtypes for all scitypes are in the tag with tagname.

        Mutates self tag with name ``tagname``.
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

        backend = self.backend
        backend_params = self.backend_params if self.backend_params else {}

        def evaluate_candidates(candidate_params):
            candidate_params = list(candidate_params)

            if self.verbose > 0:
                n_candidates = len(candidate_params)
                n_splits = cv.get_n_splits(y)
                print(
                    f"Fitting {n_splits} folds for each of {n_candidates} candidates,"
                    f" totalling {n_candidates * n_splits} fits"
                )

            # Set meta variables for parallelization.
            meta = {}
            meta["forecaster"] = self.forecaster
            meta["y"] = y
            meta["X"] = X
            meta["cv"] = cv
            meta["strategy"] = self.strategy
            meta["scoring"] = scoring
            meta["error_score"] = self.error_score
            meta["scoring_name"] = scoring_name

            out = parallelize(
                fun=_fit_and_score,
                iter=candidate_params,
                meta=meta,
                backend=backend,
                backend_params=backend_params,
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
        _forecasters_to_return = min(self.return_n_best_forecasters, len(results.index))
        if _forecasters_to_return == -1:
            _forecasters_to_return = len(results.index)
        for i in range(_forecasters_to_return):
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


def _fit_and_score(params, meta):
    """Fit and score forecaster with given parameters.

    Root level function for parallelization, called from
    BaseGridSearchCV._fit, evaluate_candidates, within parallelize.
    """
    meta = meta.copy()
    scoring_name = meta.pop("scoring_name")

    # Set parameters.
    forecaster = meta.pop("forecaster").clone()
    forecaster.set_params(**params)

    # Evaluate.
    out = evaluate(forecaster, **meta)

    # Filter columns.
    out = out.filter(items=[scoring_name, "fit_time", "pred_time"], axis=1)

    # Aggregate results.
    out = out.mean()
    out = out.add_prefix("mean_")

    # Add parameters to output table.
    out["params"] = params

    return out
