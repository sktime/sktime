# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Extension template for forecasters, SIMPLE version.

How to use this implementation template to implement a new estimator:
- work through all the "todo" comments below
- fill in code for mandatory methods, and optionally for optional methods
- do not write to reserved variables: is_fitted, _is_fitted, _X, _y, cutoff, _fh,
    _cutoff, _converter_store_y, forecasters_, _tags, _tags_dynamic, _is_vectorized
- you can add more private methods, but do not override BaseEstimator's private methods
    an easy way to be safe is to prefix your methods with "_custom"
- change docstrings for functions and the file
- ensure interface compatibility by sktime.utils.estimator_checks.check_estimator
- once complete: use as a local library, or contribute to sktime via PR
- more details:
  https://www.sktime.net/en/stable/developer_guide/add_estimators.html

Testing - required for sktime test framework and check_estimator usage:
    get default parameters for test instance(s) - get_test_params()
"""
from __future__ import annotations

__author__ = ["MBristle"]

import logging

import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon


class _AutoTSAdapter(BaseForecaster):
    """Act as adapter for the AutoTS library.

    Parameter:
    ---------
        fh (ForecastingHorizon): The forecasting horizon.
        model_list (str): The list of models to use. See docs for details.
        frequency (str): The frequency of the time series.
        prediction_interval (float): The prediction interval for the forecast.
        max_generations (int): The maximum number of
            generations for the genetic algorithm.
        no_negatives (bool): Whether negative values are allowed in the forecast.
        constraint (float): The constraint on the forecast values.
        ensemble (str): The ensemble method to use.
        initial_template (str): The initial template to use for the forecast.
        random_seed (int): The random seed for reproducibility.
        holiday_country (str): The country for holiday effects.
        subset (int): The subset of the data to use.
        aggfunc (str): The aggregation function to use.
        na_tolerance (float): The tolerance for missing values.
        metric_weighting (dict): The weights for different forecast evaluation metrics.
        drop_most_recent (int): The number of most recent data points to drop.
        drop_data_older_than_periods (int): The threshold for dropping old data points.
        transformer_list (dict): The list of transformers to use.
        transformer_max_depth (int): The maximum depth of the transformers.
        models_mode (str): The mode for selecting models.
        num_validations (int): The number of validations to perform.
        models_to_validate (float): The fraction of models to validate.
        max_per_model_class (int): The maximum number of models per class.
        validation_method (str): The method for validation.
        min_allowed_train_percent (float): The minimum percentage of
            data allowed for training.
        remove_leading_zeroes (bool): Whether to remove leading zeroes from the data.
        prefill_na (str): The method for prefilling missing values.
        introduce_na (bool): Whether to introduce missing values to the data.
        preclean (dict): The parameters for data pre-cleaning.
        model_interrupt (bool): Whether the model can be interrupted.
        generation_timeout (int): The timeout for each generation.
        current_model_file (str): The file containing the current model.
        verbose (int): The verbosity level.
        n_jobs (int): The number of jobs to run in parallel.
        model_name: (str): The name of the model.
            NOTE: Overwrites the model_list parameter. For using only one model.
    """

    _tags = {
        "scitype:y": "univariate",
        "authors": [
            "MBristle",
        ],
        # "maintainers": ["MBristle"],
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        "ignores-exogeneous-X": True,
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "requires-fh-in-fit": True,
        "reserved_params": ["fh_"],
    }

    def __init__(
        self,
        model_list: str = "default",
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
        model_name: str = "",
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
        fh: ForecastingHorizon | None = None,
        X: pd.DataFrame | None = None,
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
        # sets y_index_was_period_ and self.y_index_was_int_ flags
        # to remember the index type of y before conversion
        # self._remember_y_input_index_type(y)

        # various type input indices are converted to datetime
        # since Auto can only deal with dates
        y = self._convert_input_to_date(y)
        # We have to bring the data into the required format for fbprophet
        # the index should not be pandas index, but in a column named "ds"
        df = y.copy()
        # df.name = "y"
        # df.index.name = "ds"
        # df = df.reset_index()
        # if type(df) == pd.Series:
        #     df = df.to_frame()
        self._y = df

        self._fh = fh
        self._instantiate_model()
        try:
            self.forecaster_.fit(df=self._y)
        except Exception as e:
            raise e
            # breakpoint()
        return self

    def _predict(
        self, fh: ForecastingHorizon | None = None, X: pd.DataFrame | None = None
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

    # todo: implement this if this is an estimator contributed to sktime
    #   or to run local automated unit and integration testing of estimator
    #   method should return default parameters, so that a test instance can be created
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
        params = {
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
        }
        return params

    def _instantiate_model(self):
        """Instantiate the autoTS model.

        Returns
        -------
            self: This function returns a reference to the instantiated model object.
        """
        if type(self.ensemble) is not list:
            ensemble = [
                self.ensemble,
            ]
        else:
            ensemble = self.ensemble

        if type(self.model_interrupt) is not tuple:
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

        if self.metric_weighting is None:
            metric_weighting = (
                {
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
            )

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
        elif type(y.index) is pd.PeriodIndex:
            y = y.copy()
            y.index = y.index.to_timestamp()
        elif y.index.is_integer():
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
        if type(self._fh) is ForecastingHorizon:
            # TODO: not nice to access the private attribute (_values),
            #  consider better solution
            cutoff = self._fh_cutoff_transformation(self._y)
            fh_length = max(self._fh.to_relative(cutoff)._values)
            if fh_length <= 0:
                # breakpoint()
                raise ValueError(
                    "The relative length to the training data of "
                    "the forecasting horizon must be bigger than 0."
                )
            return fh_length
        else:
            raise NotImplementedError

    def _fh_cutoff_transformation(self, cutoff):
        if isinstance(self._fh._values, (pd.Period, pd.PeriodIndex)):
            return cutoff.index.to_period()[-1]
        elif isinstance(self._fh._values, pd.DatetimeIndex):
            return cutoff.index[-1]

        else:
            return len(cutoff.index)
