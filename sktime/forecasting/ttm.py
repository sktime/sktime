# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of TinyTimeMixer for forecasting."""

__author__ = ["geetu040"]


import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.utils.warnings import warn
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", severity="none"):
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""
        pass


# todo: change class name and write docstring
class TinyTimeMixerForecaster(_BaseGlobalForecaster):
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
        "capability:global_forecasting": True,
    }
    #  in case of inheritance, concrete class should typically set tags
    #  alternatively, descendants can set tags in __init__ (avoid this if possible)

    # todo: add any hyper-parameters and components to constructor
    def __init__(
        self,
        model_path="ibm/TTM",
        revision="main",
        config=None,
        training_args=None,
        validation_split=0.2,
        compute_metrics=None,
        callbacks=None,
    ):
        super().__init__()
        self.model_path = model_path
        self.revision = revision
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
        from tsfm_public.models.tinytimemixer import (
            TinyTimeMixerConfig,
            TinyTimeMixerForPrediction,
        )

        # Get the Configuration
        # config = TinyTimeMixerConfig()
        config = TinyTimeMixerConfig.from_pretrained(
            self.model_path,
            revision=self.revision,
        )
        # config = PatchTSTConfig.from_pretrained(
        # "ibm-granite/granite-timeseries-patchtst"
        # )

        # Update config with user provided config
        _config = config.to_dict()
        _config.update(self._config)

        # validate patches in configuration
        # context_length / num_patches == patch_length == patch_stride
        # if this condition is not satisfied in the configuration
        # this error is raised in forward pass of the model
        # RuntimeError: mat1 and mat2 shapes cannot be multiplied (384x4 and 32x64)
        context_length = _config.get("context_length")
        num_patches = _config.get("num_patches")
        patch_length = _config.get("patch_length")
        patch_stride = _config.get("patch_stride")
        patch_size = context_length / num_patches
        if patch_size != patch_length or patch_stride != patch_stride:
            # update the config here
            patch_size = max(1, int(patch_size))
            _config["patch_length"] = patch_size
            _config["patch_stride"] = patch_size
            _config["num_patches"] = _config["context_length"] // patch_size

            msg = (
                "Invalid configuration detected. "
                "The provided values do not satisfy the required condition:\n"
                "context_length / num_patches == patch_length == patch_stride\n"
                "Provided configuration:\n"
                f"- context_length: {context_length}\n"
                f"- num_patches: {num_patches}\n"
                f"- patch_length: {patch_length}\n"
                f"- patch_stride: {patch_stride}\n"
                "Configuration has been automatically updated to:\n"
                f"- context_length: {context_length}\n"
                f"- num_patches: {_config['num_patches']}\n"
                f"- patch_length: {_config['patch_length']}\n"
                f"- patch_stride: {_config['patch_stride']}"
            )
            warn(msg)

        if fh is not None:
            _config["prediction_length"] = max(
                *(fh.to_relative(self._cutoff)._values),
                _config["prediction_length"],
            )

        config = config.from_dict(_config)

        # Get the Model
        # self.model, info = PatchTSTForPrediction.from_pretrained(
        # "ibm-granite/granite-timeseries-patchtst",
        self.model, info = TinyTimeMixerForPrediction.from_pretrained(
            self.model_path,
            revision=self.revision,
            config=config,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
        )

    def _predict(self, fh, X, y=None):
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

        hist =  y if self._global_forecasting else self._y

        # get the last 'context_length' values from hist
        hist = hist[-self.model.config.context_length :].values

        # initialize 'past_values' and 'observed_mask' with the correct lengths
        past_values = np.zeros((self.model.config.context_length, hist.shape[-1]))
        observed_mask = np.zeros((self.model.config.context_length, hist.shape[-1]))

        # update 'past_values' and 'observed_mask' with '_y' and ones respectively
        past_values[-len(hist) :] = hist
        observed_mask[-len(hist) :] = 1

        # convert to torch tensors
        past_values = torch.tensor(np.expand_dims(past_values, axis=0)).float()
        observed_mask = torch.tensor(np.expand_dims(observed_mask, axis=0)).float()

        self.model.eval()
        outputs = self.model(
            past_values=past_values,
            observed_mask=observed_mask,
        )
        outputs = outputs.prediction_outputs.detach().numpy()[0]

        index = (
            ForecastingHorizon(range(1, len(outputs) + 1))
            .to_absolute(self._cutoff)
            ._values
        )
        columns = self._y.columns

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
                "config": {},
                "validation_split": 0.2,
                "training_args": {
                    "num_train_epochs": 1,
                    "output_dir": "test_output",
                    "per_device_train_batch_size": 32,
                },
            },
        ]
        return params


class PyTorchDataset(Dataset):
    """Dataset for use in sktime deep learning forecasters."""

    def __init__(
        self,
        y: pd.DataFrame,
        seq_len: int,
        fh=None,
        X: pd.DataFrame = None,
        batch_size=8,
        no_size1_batch=True,
    ):
        if not isinstance(y.index, pd.MultiIndex):
            self.y = np.array(y.values, dtype=np.float32).reshape(1, len(y), 1)
            self.X = (
                np.array(X.values, dtype=np.float32).reshape(
                    (1, len(X), len(X.columns))
                )
                if X is not None
                else X
            )
        else:
            self.y = _frame2numpy(y)
            self.X = _frame2numpy(X) if X is not None else X

        self._num, self._len, _ = self.y.shape
        self.fh = fh
        self.seq_len = seq_len
        self._len_single = self._len - self.seq_len - self.fh + 1
        self.batch_size = batch_size
        self.no_size1_batch = no_size1_batch

    def __len__(self):
        """Return length of dataset."""
        true_length = self._num * max(self._len_single, 0)
        if self.no_size1_batch and true_length % self.batch_size == 1:
            return true_length - 1
        else:
            return true_length

    def __getitem__(self, i):
        """Return data point."""
        from torch import tensor

        m = i % self._len_single
        n = i // self._len_single
        hist_y = tensor(self.y[n, m : m + self.seq_len, :]).float().flatten()
        futu_y = (
            tensor(self.y[n, m + self.seq_len : m + self.seq_len + self.fh, :])
            .float()
            .flatten()
        )
        if self.X is not None:
            exog_data = tensor(
                self.X[n, m + self.seq_len : m + self.seq_len + self.fh, :]
            ).float()
            hist_exog = tensor(self.X[n, m : m + self.seq_len, :]).float()
        else:
            exog_data = tensor([[]] * self.fh)
            hist_exog = tensor([[]] * self.seq_len)

        return {
            "past_values": hist_y,
            "past_time_features": hist_exog,
            "future_time_features": exog_data,
            "past_observed_mask": (~hist_y.isnan()).to(int),
            "future_values": futu_y,
        }
