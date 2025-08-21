# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for StatsForecast models."""

from inspect import signature
from warnings import warn

import pandas

from sktime.forecasting.base import BaseForecaster
from sktime.utils.adapters.forward import _clone_fitted_params

__all__ = ["_GeneralisedStatsForecastAdapter", "StatsForecastBackAdapter"]
__author__ = ["yarnabrina", "arnaujc91", "luca-miniati"]


class _GeneralisedStatsForecastAdapter(BaseForecaster):
    """Base adapter class for StatsForecast models."""

    _tags = {
        # packaging info
        # --------------
        "authors": ["yarnabrina", "arnaujc91"],
        "maintainers": ["yarnabrina"],
        "python_version": ">=3.8",
        # todo 0.39.0: check whether scipy<1.16 is still needed
        "python_dependencies": ["statsforecast", "scipy<1.16"],
        # estimator type
        # --------------
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "requires-fh-in-fit": False,
        # "X-y-must-have-same-index": True,  # TODO: need to check (how?)
        # "enforce_index_type": None,  # TODO: need to check (how?)
        "capability:missing_values": False,
    }

    def __init__(self):
        super().__init__()

        self._forecaster = None
        self._fitted_forecaster = None
        pred_supported = self._check_supports_pred_int()
        self._support_pred_int_in_sample = pred_supported["int_in_sample"]
        self._support_pred_int = pred_supported["int"]

        self.set_tags(
            **{"capability:pred_int:insample": self._support_pred_int_in_sample}
        )
        self.set_tags(**{"capability:pred_int": self._support_pred_int})

    def _get_statsforecast_class(self):
        raise NotImplementedError("abstract method")

    def _get_statsforecast_params(self) -> dict:
        return self.get_params()

    def _get_init_statsforecast_params(self):
        """Return parameters in __init__ statsforecast forecaster.

        Return a list of parameters in the __init__ method from
        the statsforecast forecaster class used in the sktime adapter.
        """
        statsforecast_class = self._get_statsforecast_class()
        return list(signature(statsforecast_class.__init__).parameters.keys())

    def _get_statsforecast_default_params(self) -> dict:
        """Get default parameters for the statsforecast forecaster.

        This will in general be different from self.get_param_defaults(),
        as the set or names of inner parameters can differ.

        For parameters without defaults, will use the parameter
        of self instead.
        """
        self_params = self.get_params(deep=False)
        self_default_params = self.get_param_defaults()
        self_params.update(self_default_params)
        cls_with_defaults = type(self)(**self_params)
        return cls_with_defaults._get_statsforecast_params()

    def _get_validated_statsforecast_params(self) -> dict:
        """Return parameter dict with only parameters accepted by statsforecast API.

        Checks if the parameters passed to the statsforecast forecaster
        are valid in the __init__ method of the aforementioned forecaster.
        If the parameter is not there it will just not be passed. Furthermore
        if the parameter is modified by the sktime user,
        he will be notified that the parameter does not exist
        anymore in the version installed of statsforecast by the user.

        """
        params_sktime_to_statsforecast: dict = self._get_statsforecast_params()
        params_sktime_to_statsforecast_default: dict = (
            self._get_statsforecast_default_params()
        )
        statsforecast_init_params = set(self._get_init_statsforecast_params())

        # Filter sktime_params to only include keys in statsforecast_params
        filtered_sktime_params = {
            key: value
            for key, value in params_sktime_to_statsforecast.items()
            if key in statsforecast_init_params
        }

        non_default_params = [
            p
            for p in params_sktime_to_statsforecast
            if params_sktime_to_statsforecast[p]
            != params_sktime_to_statsforecast_default[p]
        ]
        # Find parameters not in statsforecast_params or sktime_default_params
        param_diff = set(non_default_params) - statsforecast_init_params

        if param_diff:
            params_str = ", ".join([f'"{param}"' for param in param_diff])
            warning_message = (
                f"Keyword arguments {params_str} "
                f"will be omitted as they are not found in the __init__ method from "
                f"{self._get_statsforecast_class()}. Check your statsforecast version "
                f"to find out the right API parameters."
            )
            warn(warning_message)

        return filtered_sktime_params

    def _instantiate_model(self):
        cls = self._get_statsforecast_class()
        params = self._get_validated_statsforecast_params()
        return cls(**params)

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        del fh  # avoid being detected as unused by ``vulture`` like tools

        self._forecaster = self._instantiate_model()

        y_fit_input = y.to_numpy(copy=False)

        X_fit_input = X
        if X_fit_input is not None:
            X_fit_input = X.to_numpy(copy=False)

        # StatsForecast occasionally switch to a different model when fitting based on
        # the data. This means that the model is not guaranteed to be the same as the
        # one that was instantiated, and that one will be marked as un-fitted, making it
        # unsuitable for further processing. Hence, we keep track of the fitted model as
        # well, and use that exclusively from now onwards.
        # Refer to issue #7969 and PR #7983 for more details.
        self._fitted_forecaster = self._forecaster.fit(y_fit_input, X=X_fit_input)

        # clone fitted parameters to self
        _clone_fitted_params(self, self._fitted_forecaster, overwrite=False)

        return self

    def _predict_in_or_out_of_sample(self, fh, fh_type, X=None, levels=None):
        maximum_forecast_horizon = fh.to_relative(self.cutoff)[-1]

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        horizon_positions = fh.to_indexer(self.cutoff)

        level_arguments = None if levels is None else [100 * level for level in levels]

        if fh_type == "in-sample":
            predict_method = self._fitted_forecaster.predict_in_sample
            # Before v1.5.0 (from statsforecast) not all foreasters
            # have a "level" keyword argument in `predict_in_sample`
            level_kw = (
                {"level": level_arguments} if self._support_pred_int_in_sample else {}
            )
            predictions = predict_method(**level_kw)
            point_predictions = predictions["fitted"]
        elif fh_type == "out-of-sample":
            predict_method = self._fitted_forecaster.predict
            # Before v1.5.0 (from statsforecast) not all foreasters
            # have a "level" keyword argument in `predict`
            level_kw = {"level": level_arguments} if self._support_pred_int else {}
            predictions = predict_method(maximum_forecast_horizon, X=X, **level_kw)
            point_predictions = predictions["mean"]

        if isinstance(point_predictions, pandas.Series):
            point_predictions = point_predictions.to_numpy()

        final_point_predictions = pandas.Series(
            point_predictions[horizon_positions], index=absolute_horizons
        )

        if levels is None:
            return final_point_predictions

        var_names = self._get_varnames()
        var_name = var_names[0]

        interval_predictions_indices = pandas.MultiIndex.from_product(
            [var_names, levels, ["lower", "upper"]]
        )
        interval_predictions = pandas.DataFrame(
            index=absolute_horizons, columns=interval_predictions_indices
        )

        if fh_type == "out-of-sample":
            column_prefix = ""
        elif fh_type == "in-sample":
            column_prefix = "fitted-"

        for level, level_argument in zip(levels, level_arguments):
            lower_interval_predictions = predictions[
                f"{column_prefix}lo-{level_argument}"
            ]
            if isinstance(lower_interval_predictions, pandas.Series):
                lower_interval_predictions = lower_interval_predictions.to_numpy()

            upper_interval_predictions = predictions[
                f"{column_prefix}hi-{level_argument}"
            ]
            if isinstance(upper_interval_predictions, pandas.Series):
                upper_interval_predictions = upper_interval_predictions.to_numpy()

            interval_predictions[(var_name, level, "lower")] = (
                lower_interval_predictions[horizon_positions]
            )
            interval_predictions[(var_name, level, "upper")] = (
                upper_interval_predictions[horizon_positions]
            )

        return interval_predictions

    def _split_horizon(self, fh):
        in_sample_horizon = fh.to_in_sample(self.cutoff)
        out_of_sample_horizon = fh.to_out_of_sample(self.cutoff)

        return in_sample_horizon, out_of_sample_horizon

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
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        X_predict_input = X.to_numpy(copy=False) if X is not None else X

        in_sample_horizon, out_of_sample_horizon = self._split_horizon(fh)

        point_predictions = []

        if in_sample_horizon:
            in_sample_point_predictions = self._predict_in_or_out_of_sample(
                in_sample_horizon, "in-sample"
            )
            point_predictions.append(in_sample_point_predictions)

        if out_of_sample_horizon:
            out_of_sample_point_predictions = self._predict_in_or_out_of_sample(
                out_of_sample_horizon, "out-of-sample", X=X_predict_input
            )
            point_predictions.append(out_of_sample_point_predictions)

        final_point_predictions = pandas.concat(point_predictions, copy=False)
        final_point_predictions.name = self._y.name

        return final_point_predictions

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

        private _predict_interval containing the core logic,
            called from predict_interval and possibly predict_quantiles

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        coverage : list of float (guaranteed not None and floats in [0,1] interval)
           nominal coverage(s) of predictive interval(s)

        Returns
        -------
        pred_int : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level coverage fractions for which intervals were computed.
                    in the same order as in input `coverage`.
                Third level is string "lower" or "upper", for lower/upper interval end.
            Row index is fh, with additional (upper) levels equal to instance levels,
                from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are forecasts of lower/upper interval end,
                for var in col index, at nominal coverage in second col index,
                lower/upper depending on third col index, for the row index.
                Upper/lower interval end forecasts are equivalent to
                quantile forecasts at alpha = 0.5 - c/2, 0.5 + c/2 for c in coverage.
        """
        X_predict_input = X if X is None else X.to_numpy(copy=False)

        in_sample_horizon, out_of_sample_horizon = self._split_horizon(fh)

        interval_predictions = []

        if in_sample_horizon:
            in_sample_interval_predictions = self._predict_in_or_out_of_sample(
                in_sample_horizon,
                "in-sample",
                levels=coverage,
            )
            interval_predictions.append(in_sample_interval_predictions)

        if out_of_sample_horizon:
            out_of_sample_interval_predictions = self._predict_in_or_out_of_sample(
                out_of_sample_horizon,
                "out-of-sample",
                X=X_predict_input,
                levels=coverage,
            )
            interval_predictions.append(out_of_sample_interval_predictions)

        final_interval_predictions = pandas.concat(interval_predictions, copy=False)

        return final_interval_predictions

    def _check_supports_pred_int(self) -> dict[str, bool]:
        """
        Check if prediction intervals will work with forecaster.

        Check if `level` argument is available in `predict_in_sample` and
        `predict` methods from the `statsforecast` forecaster.
        A tuple of booleans (`support_pred_int_in_sample`, `support_pred_int`)
        is returned where the user is informed which of the two, if any,
        support interval predictions.

         Furthermore, will throw a warning to let the user know that he should consider
         upgrading statsforecast version as both or one of the two methods might
         not be able to produce confidence intervals.

        Returns
        -------
        Dict of bool
            A dict containing two boolean values:
            - `support_pred_int_in_sample`: True if prediction intervals are supported
              in `predict_in_sample`, False otherwise.
            - `support_pred_int`: True if prediction intervals are supported
              in `predict`, False otherwise.
        """
        try:  # try/except to avoid import errors at construction
            statsforecast_class = self._get_statsforecast_class()
        except Exception:
            return {"int_in_sample": False, "int": False}

        if (
            "level"
            not in signature(statsforecast_class.predict_in_sample).parameters.keys()
        ):
            support_pred_int_in_sample = False
            import statsforecast

            warn(
                f" {statsforecast_class.__name__} from "
                f"statsforecast v{statsforecast.__version__} "
                f"does not support prediction of intervals in `predict_in_sample`. "
                f"Consider upgrading to a newer version."
            )
        else:
            support_pred_int_in_sample = True

        if "level" not in signature(statsforecast_class.predict).parameters.keys():
            support_pred_int = False
            import statsforecast

            warn(
                f" {statsforecast_class.__name__} from "
                f"statsforecast v{statsforecast.__version__} "
                f"does not support prediction of intervals in `predict`. "
                f"Consider upgrading to a newer version."
            )
        else:
            support_pred_int = True

        return {"int_in_sample": support_pred_int_in_sample, "int": support_pred_int}


class StatsForecastBackAdapter:
    """StatsForecast Back Adapter.

    StatsForecastBackAdapter is a wrapper for sktime forecasters to be used in
    StatsForecast composite models.

    Parameters
    ----------
    estimator : sktime forecaster

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.statsforecast import StatsForecastMSTL
    >>> from sktime.forecasting.ets import AutoETS

    >>> y = load_airline()
    >>> trend_forecaster = AutoETS() # doctest: +SKIP
    >>> model = StatsForecastMSTL( # doctest: +SKIP
            season_length=[3,12],
            trend_forecaster=trend_forecaster
        )
    >>> fitted_model = model.fit(y=y) # doctest: +SKIP
    >>> y_pred = fitted_model.predict(fh=[1,2,3]) # doctest: +SKIP
    """

    _tags = {
        "python_dependencies": ["statsforecast"],
    }

    def __init__(self, estimator):
        super().__init__()

        self.estimator = estimator
        self.prediction_intervals = None

    def __repr__(self):
        """Representation dunder."""
        return "StatsForecastBackAdapter"

    def new(self):
        """Make new instance of back-adapter."""
        _self = type(self).__new__(type(self))
        _self.__dict__.update(self.__dict__)
        return _self

    def fit(self, y, X=None):
        """Fit to training data.

        Parameters
        ----------
        y : ndarray
            Time series of shape (t, ) without missing values
        X : typing.Optional[numpy.ndarray], default=None

        Returns
        -------
        self : returns an instance of self.
        """
        self.estimator = self.estimator.fit(y=y, X=X)

        return self

    def predict(self, h, X=None, level=None):
        """Make forecasts.

        Parameters
        ----------
        h : int
            Forecast horizon.
        X : typing.Optional[numpy.ndarray], default=None
            Optional exogenous of shape (h, n_x).
        level : typing.Optional[typing.Tuple[int]], default=None
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        y_pred : dict
            Dictionary with entries mean for point predictions and level_* for
            probabilistic predictions.
        """
        mean = self.estimator.predict(fh=range(1, h + 1), X=X)[:, 0]
        if level is None:
            return {"mean": mean}
        # if a level is passed, and if prediction_intervals has not been instantiated
        # yet
        elif self.prediction_intervals is None:
            from statsforecast.utils import ConformalIntervals

            self.prediction_intervals = ConformalIntervals(h=h)

        level = sorted(level)
        coverage = [round(_l / 100, 2) for _l in level]

        pred_int = self.estimator.predict_interval(
            fh=range(1, h + 1), X=X, coverage=coverage
        )

        return self.format_pred_int("mean", mean, pred_int, coverage, level)

    def predict_in_sample(self, level=None):
        """Access fitted MSTL insample predictions.

        Parameters
        ----------
        level : typing.Optional[typing.Tuple[int]]
            Confidence levels (0-100) for prediction intervals.

        Returns
        -------
        y_pred : dict
            Dictionary with entries mean for point predictions and level_* for
            probabilistic predictions.
        """
        fitted = self.estimator.predict(self.estimator._y.index)[:, 0]

        if level is None:
            return {"fitted": fitted}

        level = sorted(level)
        coverage = [round(_l / 100, 2) for _l in level]
        pred_int = self.estimator.predict_interval(
            fh=self.estimator._y.index, X=self.estimator._X, coverage=coverage
        )
        return self.format_pred_int("fitted", fitted, pred_int, coverage, level)

    def format_pred_int(self, y_pred_name, y_pred, pred_int, coverage, level):
        """Convert prediction intervals into a StatsForecast-format dictionary."""
        pred_int_prefix = "fitted-" if y_pred_name == "fitted" else ""

        pred_int_no_lev = pred_int.droplevel(0, axis=1)

        return {
            y_pred_name: y_pred,
            **{
                f"{pred_int_prefix}lo-{_l}": pred_int_no_lev[(c, "lower")].values
                for c, _l in zip(reversed(coverage), reversed(level))
            },
            **{
                f"{pred_int_prefix}hi-{_l}": pred_int_no_lev[(c, "upper")].values
                for c, _l in zip(coverage, level)
            },
        }

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        from sktime.forecasting.theta import ThetaForecaster
        from sktime.forecasting.var import VAR
        from sktime.utils.dependencies import _check_estimator_deps

        del parameter_set  # to avoid being detected as unused by ``vulture`` etc.

        stm_ests = [ThetaForecaster, VAR]
        if _check_estimator_deps(stm_ests, severity="none"):
            params = [
                {
                    "estimator": ThetaForecaster(),
                },
                {
                    "estimator": VAR(),
                },
            ]
        else:
            from sktime.forecasting.naive import NaiveForecaster

            params = [
                {
                    "estimator": NaiveForecaster(),
                },
            ]

        return params
