# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of TinyTimeMixer for forecasting."""

__author__ = ["geetu040"]


from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster, ForecastingHorizon

# todo: for imports of sktime soft dependencies:
# make sure to fill in the "python_dependencies" tag with the package import name
# import soft dependencies only inside methods of the class, not at the top of the file


# todo: change class name and write docstring
class TinyTimeMixerForecaster(BaseForecaster):
    """Custom forecaster. todo: write docstring.

    todo: describe your custom forecaster here

    Parameters
    ----------
    parama : int
        descriptive explanation of parama
    paramb : string, optional (default='default')
        descriptive explanation of paramb
    paramc : boolean, optional (default= whether paramb is not the default)
        descriptive explanation of paramc
    and so on
    est : sktime.estimator, BaseEstimator descendant
        descriptive explanation of est
    est2: another estimator
        descriptive explanation of est2
    and so on
    """

    # todo: fill out estimator tags here
    #  tags are inherited from parent class if they are not set
    # todo: define the forecaster scitype by setting the tags
    #  the "forecaster scitype" is determined by the tags
    #   scitype:y - the expected input scitype of y - univariate or multivariate or both
    #  when changing scitype:y to multivariate or both:
    #   y_inner_mtype should be changed to pd.DataFrame
    # other tags are "safe defaults" which can usually be left as-is
    _tags = {
        # to list all valid tags with description, use sktime.registry.all_tags
        #   all_tags(estimator_types="forecaster", as_dataframe=True)
        #
        # behavioural tags: internal type
        # -------------------------------
        #
        # y_inner_mtype, X_inner_mtype control which format X/y appears in
        # in the inner functions _fit, _predict, etc
        "y_inner_mtype": "pd.DataFrame",
        "X_inner_mtype": "pd.DataFrame",
        # valid values: str and list of str
        # if str, must be a valid mtype str, in sktime.datatypes.MTYPE_REGISTER
        #   of scitype Series, Panel (panel data) or Hierarchical (hierarchical series)
        #   in that case, all inputs are converted to that one type
        # if list of str, must be a list of valid str specifiers
        #   in that case, X/y are passed through without conversion if on the list
        #   if not on the list, converted to the first entry of the same scitype
        #
        # scitype:y controls whether internal y can be univariate/multivariate
        # if multivariate is not valid, applies vectorization over variables
        "scitype:y": "both",
        # valid values: "univariate", "multivariate", "both"
        #   "univariate": inner _fit, _predict, etc, receive only univariate series
        #   "multivariate": inner methods receive only series with 2 or more variables
        #   "both": inner methods can see series with any number of variables
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        #
        # ignores-exogeneous-X = does estimator ignore the exogeneous X?
        "ignores-exogeneous-X": True,
        # valid values: boolean True (ignores X), False (uses X in non-trivial manner)
        # CAVEAT: if tag is set to True, inner methods always see X=None
        #
        # requires-fh-in-fit = is forecasting horizon always required in fit?
        "requires-fh-in-fit": True,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception in fit if fh has not been passed
        #
        # X-y-must-have-same-index = can estimator handle different X/y index?
        "X-y-must-have-same-index": True,
        # valid values: boolean True (yes), False (no)
        # if True, raises exception if X.index is not contained in y.index
        #
        # enforce_index_type = index type that needs to be enforced in X/y
        "enforce_index_type": None,
        # valid values: pd.Index subtype, or list of pd.Index subtype
        # if not None, raises exception if X.index, y.index level -1 is not of that type
        #
        # handles-missing-data = can estimator handle missing data?
        "handles-missing-data": False,
        # valid values: boolean True (yes), False (no)
        # if False, raises exception if y or X passed contain missing data (nans)
        #
        # capability:insample = can forecaster make in-sample forecasts?
        "capability:insample": False,
        # valid values: boolean True (yes), False (no)
        # if False, exception raised if any forecast method called with in-sample fh
        #
        # capability:pred_int = does forecaster implement probabilistic forecasts?
        "capability:pred_int": False,
        # valid values: boolean True (yes), False (no)
        # if False, exception raised if proba methods are called (predict_interval etc)
        #
        # capability:pred_int:insample = can forecaster make in-sample proba forecasts?
        "capability:pred_int:insample": False,
        # valid values: boolean True (yes), False (no)
        # only needs to be set if capability:pred_int is True
        # if False, exception raised if proba methods are called with in-sample fh
        #
        # ----------------------------------------------------------------------------
        # packaging info - only required for sktime contribution or 3rd party packages
        # ----------------------------------------------------------------------------
        #
        # ownership and contribution tags
        # -------------------------------
        #
        # author = author(s) of th estimator
        # an author is anyone with significant contribution to the code at some point
        "authors": ["author1", "author2"],
        # valid values: str or list of str, should be GitHub handles
        # this should follow best scientific contribution practices
        # scope is the code, not the methodology (method is per paper citation)
        # if interfacing a 3rd party estimator, ensure to give credit to the
        # authors of the interfaced estimator
        #
        # maintainer = current maintainer(s) of the estimator
        # per algorithm maintainer role, see governance document
        # this is an "owner" type role, with rights and maintenance duties
        # for 3rd party interfaces, the scope is the sktime class only
        "maintainers": ["maintainer1", "maintainer2"],
        # valid values: str or list of str, should be GitHub handles
        # remove tag if maintained by sktime core team
        #
        # dependency tags: python version and soft dependencies
        # -----------------------------------------------------
        #
        # python version requirement
        "python_version": None,
        # valid values: str, PEP 440 valid python version specifiers
        # raises exception at construction if local python version is incompatible
        #
        # soft dependency requirement
        "python_dependencies": None,
        # valid values: str or list of str, PEP 440 valid package version specifiers
        # raises exception at construction if modules at strings cannot be imported
    }
    #  in case of inheritance, concrete class should typically set tags
    #  alternatively, descendants can set tags in __init__ (avoid this if possible)

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        config=None,
        training_args=None,
        validation_split=0.2,
        compute_metrics=None,
        callbacks=None,
    ):
        super().__init__()
        self.config = config
        self._config = config if config is not None else {}
        self.training_args = training_args
        self._training_args = training_args if training_args is not None else {}
        self.validation_split = validation_split
        self.compute_metrics = compute_metrics
        self.callbacks = callbacks

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
        # from transformers import PatchTSTForPrediction, PatchTSTConfig
        from transformers import Trainer, TrainingArguments
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerConfig,
            TinyTimeMixerForPrediction,
        )

        # Get the Configuration
        config = TinyTimeMixerConfig()
        # config = TinyTimeMixerConfig.from_pretrained("ibm/TTM", revision="1024_96_v1")
        # config = PatchTSTConfig.from_pretrained(
        # "ibm-granite/granite-timeseries-patchtst"
        # )

        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self._config)
        # TODO: validate this configuration
        # context_length / num_patches == patch_length == patch_stride

        if fh is not None:
            _config["prediction_length"] = max(
                *(fh.to_relative(self._cutoff)._values + 1),
                _config["prediction_length"],
            )

        config = config.from_dict(_config)

        # Get the Model
        # self.model, info = PatchTSTForPrediction.from_pretrained(
        # "ibm-granite/granite-timeseries-patchtst",
        # self.model, info = TinyTimeMixerForPrediction.from_pretrained(
        #     "ibm/TTM", revision="1024_96_v1",
        #     config=config,
        #     output_loading_info=True,
        #     ignore_mismatched_sizes=True,
        # )
        self.model = TinyTimeMixerForPrediction(config)

        # Get the Dataset
        train_dataset, eval_dataset = self._get_dataset(
            y=y,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
        )

        # Get Training Configuration
        training_args = deepcopy(self._training_args)
        training_args = TrainingArguments(**training_args)

        # Get the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        # Train the model
        trainer.train()

    def _get_dataset(self, y, context_length, prediction_length):
        target_columns = y.columns
        timestamp_column = y.index.name or "index"

        data = y.copy()
        data.index = self._handle_data_index(data.index)
        data = data.astype(float)
        data.reset_index(inplace=True)

        from tsfm_public.toolkit.time_series_preprocessor import (
            TimeSeriesPreprocessor,
            get_datasets,
        )

        tsp = TimeSeriesPreprocessor(
            target_columns=target_columns,
            timestamp_column=timestamp_column,
            context_length=context_length,
            prediction_length=prediction_length,
        )
        train_dataset, eval_dataset, _ = get_datasets(
            ts_preprocessor=tsp,
            dataset=data,
            split_config={
                "train": 1 - self.validation_split,
                "eval": self.validation_split,
                "test": 0,
            },
        )

        return train_dataset, eval_dataset

    def _handle_data_index(self, index):
        if isinstance(index, pd.DatetimeIndex):
            return index

        if isinstance(index, pd.PeriodIndex):
            return index.to_timestamp()

        if pd.api.types.is_integer_dtype(index):
            return pd.to_datetime("2021-01-01") + pd.to_timedelta(index, unit="D")

        return index

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
        import torch

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        _y = self._y[-self.model.config.context_length :]

        inputs = np.expand_dims(_y.values, axis=0)
        inputs = torch.tensor(inputs, dtype=torch.float)
        self.model.eval()
        outputs = self.model(inputs)
        outputs = outputs.prediction_outputs.detach().numpy()[0]

        index = (
            ForecastingHorizon(range(1, len(outputs) + 1))
            .to_absolute(self._cutoff)
            ._values
        )
        columns = _y.columns

        pred = pd.DataFrame(
            outputs,
            index=index,
            columns=columns,
        )
        pred = pred.loc[fh.to_absolute(self.cutoff)._values]

        return pred

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
                "config": {
                    "context_length": 4,
                    "patch_length": 2,
                    "prediction_length": 2,
                },
                "validation_split": 0.2,
                "training_args": {
                    "num_train_epochs": 1,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 32,
                },
            },
        ]
        return params
