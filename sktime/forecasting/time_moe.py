# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of Time-MoE for forecasting."""

__author__ = ["Maple728"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.warnings import warn
import torch 
from transformers import AutoModelForCausalLM

class TimeMoE(_BaseGlobalForecaster):
    """Custom forecaster. todo: write docstring.
    
    todo: describe your custom forecaster here

    Parameters
    ----------
    context_length : int, optional (default=7)
        Length of context window for generating forecasts.
    prediction_length : int, optional (default=7)
        Length of the forecast window.
    test_size : int, optional (default=168)
        Size of test dataset.
    model_size : str, optional (default='50M')
        Size of the model to use ('50M' or '200M').
    device : str, optional (default='cpu')
        Device to run the model on ('cpu' or 'cuda').
    """
   
    def __init__(
        self,
        context_length=7,
        prediction_length=7,
        test_size=168,
        model_size='50M',
        device='cpu'
    ):
        
        model = AutoModelForCausalLM.from_pretrained(
            f'Maple728/TimeMoE-{model_size}',
            device_map=device,
            trust_remote_code=True
        )
        super().__init__()
        
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.test_size = test_size
        self.model_size = model_size
        self.device = device
        self.model=model
        # Initialize the model if necessary here
        # self.model = AutoModelForCausalLM.from_pretrained("path/to/model", size=model_size)

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

        data = torch.tensor(y.values, dtype=torch.float32)
        
        all_predictions = []
        
        with torch.no_grad():
            for i in range(0, self.test_size - self.prediction_length + 1, self.prediction_length):
                # Get sequence for current window
                start_idx = len(data) - self.test_size + i - self.context_length
                sequence = data[start_idx:start_idx + self.context_length]
                sequence = sequence.unsqueeze(0)  # Add batch dimension
                
                # Normalize sequence
                mean = sequence.mean(dim=-1, keepdim=True)
                std = sequence.std(dim=-1, keepdim=True)
                normalized_sequence = (sequence - mean) / std
                
                # Generate forecast
                output = self.model.generate(
                    normalized_sequence,
                    max_new_tokens=self.prediction_length,
                    position_ids=torch.arange(sequence.size(-1), device=self.device).unsqueeze(0)  # Example for custom position IDs
                )

                
                # Denormalize predictions
                normed_preds = output[:, -self.prediction_length:]
                predictions = normed_preds * std + mean
                all_predictions.append(predictions.squeeze(0))
        
        return torch.cat(all_predictions).numpy()

    
        #     #OR 
        #     # self.model, info = TimeMoeForPrediction.from_pretrained(
        #     # self.model_path,
        #     # revision=self.revision,
        #     # config=config,
        #     # output_loading_info=True,
        #     # ignore_mismatched_sizes=True,
        #     # )
                
        # # use it when the flash-attn is available
        # # model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="auto", attn_implementation='flash_attention_2', trust_remote_code=True)

        # # Temporary dataset
        # # seqs = torch.randn(
        # #     2, context_length
        # # )  # tensor shape is [batch_size, context_length]
        # # mean, std = seqs.mean(dim=-1, keepdim=True), seqs.std(dim=-1, keepdim=True)
        # # prediction_length = fh
        # # normed_seqs = normed_seqs.view(normed_seqs.size(0), -1)
        # # output = model.generate(
        # #     normed_seqs, max_new_tokens=prediction_length
        # # )  # shape is [batch_size, 12 + 6]
        # # normed_predictions = output[:, -prediction_length:]
        # # predictions = normed_predictions * std + mean

        # # # Handle mismatched weights by freezing model parameters
        # # if info["mismatched_keys"]:
        # #     for param in self.model.parameters():
        # #         param.requires_grad = False
        # #     for key in info["mismatched_keys"]:
        # #         module = self.model
        # #         for attr in key.split(".")[:-1]:
        # #             module = getattr(module, attr)
        # #         module.weight.requires_grad = True

        # # # Splitting the dataset for training and validation
        # # y_train, y_test = temporal_train_test_split(y, test_size=self.validation_split)

        # # Create PyTorch-compatible datasets
        # # train = PyTorchDataset(
        # #     y=y_train,
        # #     context_length=config.context_length,
        # #     prediction_length=config.prediction_length,
        # # )
        # # test = PyTorchDataset(
        # #     y=y_test,
        # #     context_length=config.context_length,
        # #     prediction_length=config.prediction_length,
        # # )

        # # # Initialize training arguments
        # # training_args = TrainingArguments(**self._training_args)

        # # # Set up the Trainer
        # # trainer = Trainer(
        # #     model=self.model,
        # #     args=training_args,
        # #     train_dataset=train,
        # #     eval_dataset=test,
        # #     compute_metrics=self.compute_metrics,
        # #     callbacks=self.callbacks,
        # # )

        # # # Train the model
        # # trainer.train()

        # # Update model reference
        # # self.model = trainer.model

        # # implement here
        # # IMPORTANT: avoid side effects to y, X, fh
        # #
        # # any model parameters should be written to attributes ending in "_"
        # #  attributes set by the constructor must not be overwritten
        # #  if used, estimators should be cloned to attributes ending in "_"
        # #  the clones, not the originals should be used or fitted if needed
        # #
        # # Note: when interfacing a model that has fit, with parameters
        # #   that are not data (y, X) or forecasting-horizon-like,
        # #   but model parameters, *don't* add as arguments to fit, but treat as follows:
        # #   1. pass to constructor,  2. write to self in constructor,
        # #   3. read from self in _fit,  4. pass to interfaced_model.fit in _fit

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

        _y = _y if self._global_forecasting else self._y

        # multi-index conversion goes here
        # if isinstance(_y.index, pd.MultiIndex):
        #     hist = _frame2numpy(_y)
        # else:
        #     hist = np.expand_dims(_y.values, axis=0)

        # hist.shape: (batch_size, n_timestamps, n_cols)

        
        h = 7

        timemoe_preds_50M = self._fit(
            y=data,
            target_column='y',
            context_length=6*h,
            prediction_length=fh,
            test_size=168,
            device='cpu'
        )
        return timemoe_preds_50M
        
        
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

