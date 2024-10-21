# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implementation of TinyTimeMixer for forecasting."""

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster
from sktime.split import temporal_train_test_split
from sktime.utils.warnings import warn

from peft import get_peft_model

if _check_soft_dependencies("torch", severity="none"):
    from torch.utils.data import Dataset
else:

    class Dataset:
        """Dummy class if torch is unavailable."""


if _check_soft_dependencies("transformers", severity="none"):
    from transformers import Trainer, TrainingArguments


class PEFTForecaster(_BaseGlobalForecaster):
    """
    Peft Forecaster.

    Parameters
    ----------
    model : BasePeftable
    peft_config : dict
    """

    _tags = {
        "X_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "y_inner_mtype": [
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "authors": ["ajati", "wgifford", "vijaye12", "geetu040"],
        # ajati, wgifford, vijaye12 for ibm-granite code
        "maintainers": ["geetu040"],
        "python_dependencies": ["transformers", "torch"],
        "capability:global_forecasting": True,
    }

    def __init__(
        self,
        model,
        peft_config,
        training_args=None,
        callbacks=None,
        compute_metrics=None,
        device="cpu",
    ):
        super().__init__()
        self.model = model
        self.peft_config = peft_config
        self.training_args = training_args
        self.callbacks = callbacks
        self.compute_metrics = compute_metrics
        self.device = device


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
        if not self.model.is_fitted:
            self.model.fit(y, X, fh)
            
        nn_model = self.model.get_model(fh)

        train_dataset, eval_dataset = self.model.get_train_dataset(y, X, fh)


        self._peft_model = get_peft_model(
            nn_model,
            self.peft_config,
        ).to(self.device)


        # Get Training Configuration
        training_args = TrainingArguments(**self.training_args)

        # Get the Trainer
        trainer = Trainer(
            model=self._peft_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=self.callbacks,
        )

        # Train the model
        trainer.train()

        self.model.set_model(self._peft_model)


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
        return self.model.predict(fh, X, y)

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

        return [{}]

