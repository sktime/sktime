# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements and interface to the AutoTS library main entry point."""

__author__ = ["MBristle"]

import logging
from typing import Union

import pandas as pd
from pandas.api.types import is_integer_dtype

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class AutoTS(BaseForecaster):
    """Auto-ensemble from autots library by winedarksea.

    Direct interface to ``autots.AutoTS``.

    Parameters
    ----------
    model_name : str, optional (default="fast")
        The name of the model. NOTE: Overwrites the model_list parameter.
        For using only one model oder a default model_list.
    model_list : str
        The list of models to use.
        str alias or list of names of model objects to use now can be a dictionary
        of {"model": prob} but only affects starting random templates.
        Genetic algorithm takes from there.
    frequency : str
        'infer' or a specific pandas datetime offset. Can be used to force rollup of
        data (ie daily input, but frequency 'M' will rollup to monthly).
    prediction_interval: float
        0-1, uncertainty range for upper and lower forecasts.
        Adjust range, but rarely matches actual containment.
    max_generations: int
        The maximum number of generations for the genetic algorithm.
        number of genetic algorithms generations to run. More runs = longer runtime,
        generally better accuracy. It's called max because someday there will be an
        auto early stopping option, but for now this is just the exact number of
        generations to run.
    no_negatives (bool):
        Whether negative values are allowed in the forecast.
        if True, all negative predictions are rounded up to 0.
    constraint (float):
        The constraint on the forecast values.
        when not None, use this float value * data st dev above max or below min for
        constraining forecast values. now also instead accepts a
        dictionary containing the following key/values:
        constraint_method (str): one of

        * stdev_min - threshold is min and max of historic data +/- constraint
        * st dev of data stdev - threshold is the mean of historic data +/-
            constraint
        * st dev of data absolute - input is array of length series containing
            the threshold's final value for each quantile - constraint is the
            quantile of historic data to use as threshold

    constraint_regularization (float): 0 to 1
        where 0 means no constraint, 1 is hard threshold cutoff, and in between is
        penalty term
        upper_constraint (float): or array, depending on method, None if unused
        lower_constraint (float): or array, depending on method, None if unused
        bounds (bool): if True, apply to upper/lower forecast, otherwise
        False applies only to forecast
    ensemble: str
        The ensemble method to use.
        None or list or comma-separated string containing:
        'auto', 'simple', 'distance', 'horizontal', 'horizontal-min',
        'horizontal-max', "mosaic", "subsample"
    initial_template (str):
        The initial template to use for the forecast.
        'Random' - randomly generates starting template, 'General' uses template
        included in package, 'General+Random' - both of previous.
        Also can be overridden with import_template()
    random_seed (int):
        The random seed for reproducibility.
        Random seed allows (slightly) more consistent results.
    holiday_country (str):
        The country for holiday effects.
        Can be passed through to Holidays package for some models.
    subset (int):
        Maximum number of series to evaluate at once. Useful to speed evaluation
        when many series are input. takes a new subset of columns on each
        validation, unless mosaic ensembling, in which case columns are the
        same in each validation
    aggfunc (str):
        The aggregation function to use.
        If data is to be rolled up to a higher frequency (daily -> monthly) or
        duplicate timestamps are included. Default 'first' removes duplicates,
        for rollup try 'mean' or np.sum. Beware numeric aggregations like 'mean'
        will not work with non-numeric inputs. Numeric aggregations like 'sum'
        will also change nan values to 0
    na_tolerance (float):
        The tolerance for missing values.
        0 to 1. Series are dropped if they have more than this percent NaN.
        0.95 here would allow series containing up to 95% NaN values.
    metric_weighting (dict):
        The weights for different forecast evaluation metrics.
        Weights to assign to metrics, effecting how the ranking score is generated.
    drop_most_recent (int):
        Option to drop n most recent data points. Useful, say, for monthly sales
        data where the current (unfinished) month is included. occurs after any
        aggregation is applied, so will be whatever is specified by frequency,
        will drop n frequencies
    drop_data_older_than_periods (int):
        The threshold for dropping old data points.
        Will take only the n most recent timestamps.
    transformer_list (dict):
        List of transformers to use, or dict of transformer:probability.
        Note this does not apply to initial templates. can accept string aliases:
        "all", "fast", "superfast", 'scalable' (scalable is a subset of fast that
        should have fewer memory issues at scale)
    transformer_max_depth (int):
        maximum number of sequential transformers to generate for new Random
        Transformers. Fewer will be faster.
    models_mode (str):
        The mode for selecting models.
        option to adjust parameter options for newly generated models.
        Only sporadically utilized. Currently includes: 'default'/'random',
        'deep' (searches more params, likely slower), and 'regressor'
        (forces 'User' regressor mode in regressor capable models),
        'gradient_boosting', 'neuralnets' (~Regression class models only)
    num_validations (int):
        The number of validations to perform.
        0 for just train/test on best
        split. Possible confusion: num_validations is the number of
        validations to perform after the first eval segment, so totally
        eval/validations will be this + 1. Also "auto" and "max"
        aliases available. Max maxes out at 50.
    models_to_validate (float):
        The fraction of models to validate.
        Top n models to pass through to cross validation.
        Or float in 0 to 1 as % of tried. 0.99 is forced to 100% validation.
        1 evaluates just 1 model. If horizontal or mosaic ensemble, then
        additional min per_series models above the number here are added to
        validation.
    max_per_model_class (int):
        The maximum number of models per class. of the models_to_validate what is
        the maximum to pass from any one model class/family.
    validation_method (str):
        The method for validation.  'even', 'backwards', or 'seasonal n' where n is
        an integer of seasonal 'backwards' is better for recency and for shorter
        training sets 'even' splits the data into equally-sized slices best for
        more consistent data, a poetic but less effective strategy than others here
        'seasonal' most similar indexes 'seasonal n' for example 'seasonal 364'
        would test all data on each previous year of the forecast_length that would
        immediately follow the training data. 'similarity' automatically finds the
        data sections most similar to the most recent data that will be used for
        prediction 'custom' - if used, .fit() needs validation_indexes passed - a
        list of pd.DatetimeIndex's, tail of each is used as test
    min_allowed_train_percent (float):
        The minimum percentage of data allowed for training.
        percent of forecast length to allow as min training, else raises error.
        0.5 with a forecast length of 10 would mean 5 training points are mandated,
        for a total of 15 points. Useful in (unrecommended) cases where forecast_
        length > training length.
    remove_leading_zeroes (bool):
        Whether to remove leading zeroes from the data.
        replace leading zeroes with NaN. Useful in data where initial
        zeroes mean data collection hasn't started yet.
    prefill_na (str):
        The method for prefilling missing values.
        The value to input to fill all NaNs with. Leaving as None and allowing model
        interpolation is recommended. None, 0, 'mean', or 'median'. 0 may be useful
        in for examples sales cases where all NaN can be assumed equal to zero.
    introduce_na (bool):
        Whether to introduce missing values to the data.
        whether to force last values in one training validation to be NaN.
        Helps make more robust models. defaults to None, which introduces NaN in
        last rows of validations if any NaN in tail of training data. Will not
        introduce NaN to all series if subset is used. if True, will also randomly
        change 20% of all rows to NaN in the validations
    preclean (dict):
        The parameters for data pre-cleaning.
        if not None, a dictionary of Transformer params to be applied to input data
        {"fillna": "median", "transformations": {}, "transformation_params": {}}
        This will change data used in model inputs for fit and predict, and for
        accuracy evaluation in cross validation!
    model_interrupt (bool):
        Whether the model can be interrupted.
        If False, KeyboardInterrupts quit entire program.
        if True, KeyboardInterrupts attempt to only quit current model.
        if True, recommend use in conjunction with verbose > 0 and result_file in
        the event of accidental complete termination. if "end_generation", as True
        and also ends entire generation of run. Note skipped models will not be
        tried again.
    generation_timeout (int):
        The timeout for each generation. if not None, this is the number of minutes
        from start at which the generational search ends, then proceeding to
        validation This is only checked after the end of each generation, so
        only offers an 'approximate' timeout for searching. It is an overall
        cap for total generation search time, not per generation.
    current_model_file (str):
        The file containing the current model.
        file path to write to disk of current model params
        (for debugging if computer crashes). .json is appended
    verbose (int):
        The verbosity level.
        setting to 0 or lower should reduce most output.
        Higher numbers give more output.
    n_jobs (int):
        The number of jobs to run in parallel.
        Number of cores available to pass to parallel processing.
        A joblib context manager can be used instead (pass None in this case).
        Also 'auto'.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["winedarksea", "MBristle"],  # winedarksea for autots library
        "maintainers": ["MBristle"],
        "python_dependencies": ["autots", "pandas", "statsmodels", "scipy"],
        "python_version": ">=3.6",
        # estimator type
        # --------------
        "scitype:y": "both",
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,  # TODO: add capability
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:pred_int": False,  # TODO: add capability
        "requires-fh-in-fit": True,
    }

    def __init__(
        self,
        model_name: str = "",
        model_list: list = "superfast",
        frequency: str = "infer",
        prediction_interval: float = 0.9,
        max_generations: int = 10,
        no_negatives: bool = False,
        constraint: float = None,
        ensemble: str = "auto",
        initial_template: str = "General+Random",
        random_seed: int = 2022,
        holiday_country: str = "US",
        subset: int = None,
        aggfunc: str = "first",
        na_tolerance: float = 1,
        metric_weighting: dict = None,
        drop_most_recent: int = 0,
        drop_data_older_than_periods: int = 100000,
        transformer_list: dict = "auto",
        transformer_max_depth: int = 6,
        models_mode: str = "random",
        num_validations: int = "auto",
        models_to_validate: float = 0.15,
        max_per_model_class: int = None,
        validation_method: str = "backwards",
        min_allowed_train_percent: float = 0.5,
        remove_leading_zeroes: bool = False,
        prefill_na: str = None,
        introduce_na: bool = None,
        preclean: dict = None,
        model_interrupt: bool = True,
        generation_timeout: int = None,
        current_model_file: str = None,
        verbose: int = 1,
        n_jobs: int = -2,
    ):
        self.model_name = model_name
        self.model_list = model_list
        self.frequency = frequency
        self.prediction_interval = prediction_interval
        self.max_generations = max_generations
        self.no_negatives = no_negatives
        self.constraint = constraint
        self.ensemble = ensemble
        self.initial_template = initial_template
        self.random_seed = random_seed
        self.holiday_country = holiday_country
        self.subset = subset
        self.aggfunc = aggfunc
        self.na_tolerance = na_tolerance
        self.metric_weighting = metric_weighting
        self.drop_most_recent = drop_most_recent
        self.drop_data_older_than_periods = drop_data_older_than_periods
        self.num_validations = num_validations
        self.models_to_validate = models_to_validate
        self.validation_method = validation_method
        self.min_allowed_train_percent = min_allowed_train_percent
        self.remove_leading_zeroes = remove_leading_zeroes
        self.transformer_list = transformer_list
        self.transformer_max_depth = transformer_max_depth
        self.models_mode = models_mode
        self.max_per_model_class = max_per_model_class
        self.prefill_na = prefill_na
        self.introduce_na = introduce_na
        self.preclean = preclean
        self.model_interrupt = model_interrupt
        self.generation_timeout = generation_timeout
        self.current_model_file = current_model_file
        self.verbose = verbose
        self.n_jobs = n_jobs

        # leave this as is
        super().__init__()

        # import inside method to avoid hard dependency
        from autots import AutoTS as _autots

        self._ModelClass = _autots

    def _fit(
        self,
        y: pd.DataFrame,
        fh: Union[ForecastingHorizon, None] = None,
        X: Union[pd.DataFrame, None] = None,  # noqa: F841
    ):
        """Fits the model to the provided data.

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : pd.DataFrame
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column
            if self.get_tag("scitype:y")=="both": no restrictions apply
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead  to predict.
            Required (non-optional) here.
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        # various type input indices are converted to datetime
        # since AutoTS can only deal with dates
        y = self._convert_input_to_date(y)
        self._y = y

        self._fh = fh
        self._instantiate_model()
        try:
            self.forecaster_.fit(df=self._y)
        except Exception as e:
            raise e
        return self

    def _predict(
        self,
        fh: Union[ForecastingHorizon, None] = None,
        X: [pd.DataFrame, None] = None,  # noqa: F841
    ):
        """Provide forecast at future horizon using fitted forecaster.

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions
        """
        if fh is not None:
            self._fh = fh

        values = self.forecaster_.predict(
            forecast_length=self._get_forecast_length()
        ).forecast.values

        cutoff = self._fh_cutoff_transformation(self._y)
        values = values[self._fh.to_relative(cutoff)._values - 1]

        # convert back to original index
        row_idx: pd.Index = self._fh.to_absolute_index(self.cutoff)
        col_idx = self._y.columns
        y_pred = pd.DataFrame(values, index=row_idx, columns=col_idx)

        return y_pred

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
        params = [
            {
                "model_name": None,
                "model_list": "superfast",
                "frequency": "infer",
                "prediction_interval": 0.9,
                "max_generations": 1,
                "no_negatives": False,
                "constraint": None,
                "ensemble": "auto",
                "initial_template": "General+Random",
                "random_seed": 2022,
                "holiday_country": "US",
                "subset": None,
                "aggfunc": "first",
                "na_tolerance": 1,
                "metric_weighting": {
                    "smape_weighting": 5,
                    "mae_weighting": 2,
                    "rmse_weighting": 2,
                    "made_weighting": 0.5,
                    "mage_weighting": 0,
                    "mle_weighting": 0,
                    "imle_weighting": 0,
                    "spl_weighting": 3,
                    "containment_weighting": 0,
                    "contour_weighting": 1,
                    "runtime_weighting": 0.05,
                    "oda_weighting": 0.001,
                },
                "drop_most_recent": 0,
                "drop_data_older_than_periods": 100000,
                "transformer_list": "auto",
                "transformer_max_depth": 6,
                "models_mode": "random",
                "num_validations": "auto",
                "models_to_validate": 1,
                "max_per_model_class": None,
                "validation_method": "seasonal 1",
                "min_allowed_train_percent": 0.5,
                "remove_leading_zeroes": False,
                "prefill_na": None,
                "introduce_na": None,
                "preclean": None,
                "model_interrupt": True,
                "generation_timeout": None,
                "current_model_file": None,
                "verbose": -2,
                "n_jobs": -2,
            },
            {
                "model_list": None,
                "frequency": "infer",
                "prediction_interval": 0.9,
                "max_generations": 1,
                "no_negatives": False,
                "constraint": None,
                "ensemble": "auto",
                "initial_template": "General+Random",
                "random_seed": 2022,
                "holiday_country": "US",
                "subset": None,
                "aggfunc": "first",
                "na_tolerance": 1,
                "metric_weighting": {
                    "smape_weighting": 5,
                    "mae_weighting": 2,
                    "rmse_weighting": 2,
                    "made_weighting": 0.5,
                    "mage_weighting": 0,
                    "mle_weighting": 0,
                    "imle_weighting": 0,
                    "spl_weighting": 3,
                    "containment_weighting": 0,
                    "contour_weighting": 1,
                    "runtime_weighting": 0.05,
                    "oda_weighting": 0.001,
                },
                "drop_most_recent": 0,
                "drop_data_older_than_periods": 100000,
                "transformer_list": "auto",
                "transformer_max_depth": 6,
                "models_mode": "random",
                "num_validations": "auto",
                "models_to_validate": 1,
                "max_per_model_class": None,
                "validation_method": "seasonal 1",
                "min_allowed_train_percent": 0.5,
                "remove_leading_zeroes": False,
                "prefill_na": None,
                "introduce_na": None,
                "preclean": None,
                "model_interrupt": True,
                "generation_timeout": None,
                "current_model_file": None,
                "verbose": -1,
                "n_jobs": -2,
                "model_name": "GLS",
            },
        ]
        return params

    def _instantiate_model(self):
        """Instantiate the autoTS model.

        Returns
        -------
            self: This function returns a reference to the instantiated model object.
        """
        if not isinstance(self.ensemble, list):
            ensemble = [
                self.ensemble,
            ]
        else:
            ensemble = self.ensemble

        if not isinstance(self.model_interrupt, tuple):
            model_interrupt = (self.model_interrupt,)
        else:
            model_interrupt = self.model_interrupt

        if self.model_name is not None and self.model_name != "":
            logging.warning(
                "The parameter model_name overwrites the model_list parameter."
            )
            model_list = [self.model_name]
        else:
            model_list = self.model_list

        if not self.metric_weighting:
            metric_weighting = {
                "smape_weighting": 5,
                "mae_weighting": 2,
                "rmse_weighting": 2,
                "made_weighting": 0.5,
                "mage_weighting": 0,
                "mle_weighting": 0,
                "imle_weighting": 0,
                "spl_weighting": 3,
                "containment_weighting": 0,
                "contour_weighting": 1,
                "runtime_weighting": 0.05,
                "oda_weighting": 0.001,
            }
        else:
            metric_weighting = self.metric_weighting

        self.forecaster_ = self._ModelClass(
            model_list=model_list,
            forecast_length=self._get_forecast_length(),
            frequency=self.frequency,
            prediction_interval=self.prediction_interval,
            max_generations=self.max_generations,
            no_negatives=self.no_negatives,
            constraint=self.constraint,
            ensemble=ensemble,
            initial_template=self.initial_template,
            random_seed=self.random_seed,
            holiday_country=self.holiday_country,
            subset=self.subset,
            aggfunc=self.aggfunc,
            na_tolerance=self.na_tolerance,
            metric_weighting=metric_weighting,
            drop_most_recent=self.drop_most_recent,
            drop_data_older_than_periods=self.drop_data_older_than_periods,
            num_validations=self.num_validations,
            models_to_validate=self.models_to_validate,
            validation_method=self.validation_method,
            min_allowed_train_percent=self.min_allowed_train_percent,
            remove_leading_zeroes=self.remove_leading_zeroes,
            transformer_list=self.transformer_list,
            transformer_max_depth=self.transformer_max_depth,
            models_mode=self.models_mode,
            max_per_model_class=self.max_per_model_class,
            prefill_na=self.prefill_na,
            introduce_na=self.introduce_na,
            preclean=self.preclean,
            model_interrupt=model_interrupt,
            generation_timeout=self.generation_timeout,
            current_model_file=self.current_model_file,
            verbose=self.verbose,
            n_jobs=self.n_jobs,
        )
        return self

    def _convert_input_to_date(self, y):
        """
        Coerce y.index to pd.DatetimeIndex, for use by AutoAF.

        Parameters
        ----------
            y (pd.DataFrame or None): The input data.

        Returns
        -------
            pd.DataFrame or None: The input data with y.index coerced to
            pd.DatetimeIndex, or None if y is None.
        """
        if y is None:
            return None
        elif isinstance(y.index, pd.PeriodIndex):
            y = y.copy()
            y.index = y.index.to_timestamp()
        elif is_integer_dtype(y.index):
            y = self._convert_int_to_date(y)
        # else y is pd.DatetimeIndex as AutoAF expects, and needs no conversion
        return y

    def _convert_int_to_date(self, y):
        """
        Convert int to date, for use by AutoAF.

        Parameters
        ----------
            y (pandas.Series): The input series containing integer values.

        Returns
        -------
            pandas.Series: The input series with the index converted to dates.
        """
        y = y.copy()
        idx_max = y.index[-1] + 1
        int_idx = pd.date_range(start="2000-01-01", periods=idx_max, freq="D")
        int_idx = int_idx[y.index]
        y.index = int_idx
        return y

    def _get_forecast_length(self):
        cutoff = self._fh_cutoff_transformation(self._y)
        fh_length = max(self._fh.to_relative(cutoff)._values)
        if fh_length <= 0:
            raise ValueError(
                "The relative length to the training data of "
                "the forecasting horizon must be bigger than 0."
            )
        return fh_length

    def _fh_cutoff_transformation(self, cutoff):
        if isinstance(self._fh._values, (pd.Period, pd.PeriodIndex)):
            transformed_fh_cutoff = cutoff.index.to_period()[-1]
        elif isinstance(self._fh._values, pd.DatetimeIndex):
            transformed_fh_cutoff = cutoff.index[-1]

        else:
            transformed_fh_cutoff = len(cutoff.index)
        return transformed_fh_cutoff
