# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of Time-MoE for forecasting."""

__author__ = ["Maple728"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.sktime.forecasting.chronos import _frame2numpy
from sktime.split import temporal_train_test_split
from sktime.utils.warnings import warn

if _check_soft_dependencies("torch", severity="none"):
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


if _check_soft_dependencies("transformers", severity="none"):
    from transformers import Trainer, TrainingArguments


class TimeMoE(_BaseGlobalForecaster):
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
        "y_inner_mtype": "pd.Series",
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
        "scitype:y": "univariate",
        # valid values: "univariate", "multivariate", "both"
        #   "univariate": inner _fit, _predict, etc, receive only univariate series
        #   "multivariate": inner methods receive only series with 2 or more variables
        #   "both": inner methods can see series with any number of variables
        #
        # capability tags: properties of the estimator
        # --------------------------------------------
        #
        # ignores-exogeneous-X = does estimator ignore the exogeneous X?
        "ignores-exogeneous-X": False,
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
        "capability:insample": True,
        # valid values: boolean True (yes), False (no)
        # if False, exception raised if any forecast method called with in-sample fh
        #
        # capability:pred_int = does forecaster implement probabilistic forecasts?
        "capability:pred_int": False,
        # valid values: boolean True (yes), False (no)
        # if False, exception raised if proba methods are called (predict_interval etc)
        #
        # capability:pred_int:insample = can forecaster make in-sample proba forecasts?
        "capability:pred_int:insample": True,
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
        model_path="Maple728/TimeMoE-200M",
        revision="main",
        validation_split=0.2,
        config=None,
        training_args=None,
        compute_metrics=None,
        callbacks=None,
        broadcasting=False,
        use_source_package=False,
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
        self.broadcasting = broadcasting
        self.use_source_package = use_source_package

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.DataFrame",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : sktime time series object
            Time series to which to fit the forecaster.
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to predict.
        X : sktime time series object, optional (default=None)
            Exogenous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        if self.use_source_package:
            from sktime.libs.timemoe import TimeMoeConfig, TimeMoeForPrediction
        elif _check_soft_dependencies("torch", severity="error"):
            from sktime.libs.timemoe import TimeMoeConfig, TimeMoeForPrediction

        # Initialize model config
        config = TimeMoeConfig.from_pretrained(self.model_path, revision=self.revision)
        _config = config.to_dict()
        _config.update(self._config)

        # Ensure configuration for patching aligns with context length
        context_length = _config.get("context_length")
        num_patches = _config.get("num_patches")
        patch_length = _config.get("patch_length")
        patch_stride = _config.get("patch_stride")

        patch_size = context_length / num_patches
        if not (patch_size == patch_length == patch_stride):
            # Update the config to match patching requirements
            patch_size = max(1, int(patch_size))
            _config["patch_length"] = patch_size
            _config["patch_stride"] = patch_size
            _config["num_patches"] = _config["context_length"] // patch_size
            warn(
                "Invalid patch configuration detected. Configuration updated to ensure:\n"
                f"- context_length: {context_length}\n"
                f"- num_patches: {_config['num_patches']}\n"
                f"- patch_length and patch_stride: {patch_size}"
            )

        if fh is not None:
            _config["prediction_length"] = max(
                fh.to_relative(self._cutoff)._values,
                _config.get("prediction_length", 1),
            )

        config = TimeMoeConfig.from_dict(_config)

        # Load the model with updated config
        self.model, info = TimeMoeForPrediction.from_pretrained(
            self.model_path,
            revision=self.revision,
            config=config,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
        )

        # Handle mismatched weights by freezing model parameters
        if info["mismatched_keys"]:
            for param in self.model.parameters():
                param.requires_grad = False
            for key in info["mismatched_keys"]:
                module = self.model
                for attr in key.split(".")[:-1]:
                    module = getattr(module, attr)
                module.weight.requires_grad = True

        # Splitting the dataset for training and validation
        y_train, y_test = temporal_train_test_split(y, test_size=self.validation_split)

        # Create PyTorch-compatible datasets
        train = PyTorchDataset(
            y=y_train,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
        )
        test = PyTorchDataset(
            y=y_test,
            context_length=config.context_length,
            prediction_length=config.prediction_length,
        )

        # Initialize training arguments
        training_args = TrainingArguments(**self._training_args)

        # Set up the Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train,
            eval_dataset=test,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        # Train the model
        trainer.train()

        # Update model reference
        self.model = trainer.model

        # master_addr = os.getenv('MASTER_ADDR', '127.0.0.1')
        # master_port = os.getenv('MASTER_PORT', 9899)
        # world_size = int(os.getenv('WORLD_SIZE') or 1)
        # rank = int(os.getenv('RANK') or 0)
        # local_rank = int(os.getenv('LOCAL_RANK') or 0)
        # if torch.cuda.is_available():
        #     try:
        #         dist.init_process_group("nccl", init_method='tcp://{}:{}'.format(master_addr, master_port), rank=rank,
        #                     world_size=world_size)
        #         device = f"cuda:{local_rank}"
        #         is_dist = True
        #     except Exception as e:
        #         print('Error: ', f'Setup nccl fail, so set device to cpu: {e}')
        #         device = 'cpu'
        #         is_dist = False
        # else:
        #     device = 'cpu'
        #     is_dist = False

        # implement here
        # IMPORTANT: avoid side effects to y, X, fh
        #
        # any model parameters should be written to attributes ending in "_"
        #  attributes set by the constructor must not be overwritten
        #  if used, estimators should be cloned to attributes ending in "_"
        #  the clones, not the originals should be used or fitted if needed
        #
        # Note: when interfacing a model that has fit, with parameters
        #   that are not data (y, X) or forecasting-horizon-like,
        #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        #   1. pass to constructor,  2. write to self in constructor,
        #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

    # todo: implement this, mandatory
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
        # model = self.model
        # device = self.device
        # prediction_length = self.prediction_length

        # outputs = model.generate(
        #     inputs=batch["inputs"].to(device).to(model.dtype),
        #     max_new_tokens=prediction_length,
        # )
        # preds = outputs[:, -prediction_length:]
        # labels = batch["labels"].to(device)
        # if len(preds.shape) > len(labels.shape):
        #     labels = labels[..., None]
        # return preds, labels

        import torch

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        _y = y if self._global_forecasting else self._y

        # multi-index conversion goes here
        if isinstance(_y.index, pd.MultiIndex):
            hist = _frame2numpy(_y)
        else:
            hist = np.expand_dims(_y.values, axis=0)

        # hist.shape: (batch_size, n_timestamps, n_cols)

        # truncate or pad to match sequence length
        past_values, observed_mask = _pad_truncate(
            hist, self.model.config.context_length
        )

        past_values = (
            torch.tensor(past_values).to(self.model.dtype).to(self.model.device)
        )
        observed_mask = (
            torch.tensor(observed_mask).to(self.model.dtype).to(self.model.device)
        )

        self.model.eval()
        outputs = self.model(
            past_values=past_values,
            observed_mask=observed_mask,
        )
        pred = outputs.prediction_outputs.detach().cpu().numpy()

        # converting pred datatype

        if isinstance(_y.index, pd.MultiIndex):
            ins = np.array(
                list(np.unique(_y.index.droplevel(-1)).repeat(pred.shape[1]))
            )
            ins = [ins[..., i] for i in range(ins.shape[-1])] if ins.ndim > 1 else [ins]

            idx = (
                ForecastingHorizon(range(1, pred.shape[1] + 1), freq=self.fh.freq)
                .to_absolute(self._cutoff)
                ._values.tolist()
                * pred.shape[0]
            )
            index = pd.MultiIndex.from_arrays(
                ins + [idx],
                names=_y.index.names,
            )
        else:
            index = (
                ForecastingHorizon(range(1, pred.shape[1] + 1))
                .to_absolute(self._cutoff)
                ._values
            )

        pred = pd.DataFrame(
            # batch_size * num_timestams, n_cols
            pred.reshape(-1, pred.shape[-1]),
            index=index,
            columns=_y.columns,
        )

        absolute_horizons = fh.to_absolute_index(self.cutoff)
        dateindex = pred.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        pred = pred.loc[dateindex]
        pred.index.names = _y.index.names

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

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"est": value3, "parama": value4}
        # return params


def _pad_truncate(data, length):
    """Pad or truncate the data to the specified length."""
    if len(data) > length:
        return data[:length], [1] * length
    else:
        padding = length - len(data)
        return data + [0] * padding, [1] * len(data) + [0] * padding
