# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements an adapter for the LagLlama estimator for intergration into sktime."""

__author__ = ["shlok191"]

import subprocess

from sktime.forecasting.base import _BaseGlobalForecaster


class LagLlamaForecaster(_BaseGlobalForecaster):
    """Base class that interfaces the LagLlama forecaster.

    Parameters
    ----------
    huggingface_id : str, optional (default="time-series-foundation-models/Lag-Llama")'
        The ID of the weights for the LagLlama estimator to fetch.

    device : str, optional (default="cpu")
        Specifies the device on which to load the model.

    context_length: int, optional (default=32)
        The number of prior timestep data entries provided.

    prediction_length: int, optional (default=100)
        The length of future of timesteps to predict
        (synonymous to forecast horizon).

    num_samples: int, optional (default=10)
        Number of sample paths desired for evaluation.

    batch_size: int, optional (default=32)
        The number of batches to train for in parallel.

    nonnegative_pred_samples: bool, optional (default=False)
        If True, ensures all predicted samples are passed
        through ReLU,and are thus positive or 0.

    lr: float, optional (default=5e-5)
        The learning rate of the model.

    trainer_kwargs: dict, optional (default={"num_epochs": 50})
        The arguments to pass to the GluonTS trainer.
    """

    _tags = {
        "y_inner_mtype": "gluonts_PandasDataset_panel",
        "X_inner_mtype": "gluonts_PandasDataset_panel",
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "authors": ["shlok191"],
        "maintainers": ["shlok191"],
        "python_version": None,
        "python_dependencies": ["gluonts", "huggingface_hub", "lag_llama"],
    }

    def __init__(
        self,
        huggingface_id=None,
        device=None,
        context_length=None,
        prediction_length=None,
        num_samples=None,
        batch_size=None,
        nonnegative_pred_samples=None,
        lr=None,
        trainer_kwargs=None,
    ):
        import torch
        from lag_llama.gluon.estimator import LagLlamaEstimator

        # Defining private variable values
        self.huggingface_id = huggingface_id
        self.huggingface_id_ = (
            "time-series-foundation-models/Lag-Llama"
            if not huggingface_id
            else huggingface_id
        )

        self.device = device
        self.device_ = torch.device("cpu") if not device else device

        self.context_length = context_length
        self.context_length_ = 32 if not context_length else context_length

        self.prediction_length = prediction_length
        self.prediction_length_ = 100 if not prediction_length else prediction_length

        self.num_samples = num_samples
        self.num_samples_ = 10 if not num_samples else num_samples

        self.batch_size = batch_size
        self.batch_size_ = 32 if not batch_size else batch_size

        # Now storing the training related variables
        self.lr = lr
        self.lr_ = 5e-5 if not lr else lr

        self.trainer_kwargs = trainer_kwargs
        self.trainer_kwargs_ = (
            {"max_epochs": 50} if not trainer_kwargs else trainer_kwargs
        )

        # Not storing private variables for boolean specific values
        self.nonnegative_pred_samples = nonnegative_pred_samples

        super().__init__()

        # Downloading the LagLlama weights from Hugging Face
        download_command = f"huggingface-cli download {self.huggingface_id_} "
        +"lag-llama.ckpt --local-dir ."

        status = subprocess.run(
            download_command, shell=True, check=True, capture_output=True
        )

        # Checking if the command ran successfully
        if status.returncode != 0:
            raise RuntimeError(
                "Failed to fetch the pretrained model weights from HuggingFace!"
            )

        # Load in the lag llama checkpoint
        ckpt = torch.load("./lag-llama.ckpt", map_location=self.device_)

        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        self.estimator_args = estimator_args

        # By default, we maintain RoPE scaling
        # We provide the user an option to disable in fit() function
        rope_scaling_arguments = {
            "type": "linear",
            "factor": max(
                1.0,
                (self.context_length_ + self.prediction_length_)
                / estimator_args["context_length"],
            ),
        }

        # Creating our LagLlama estimator
        self.estimator_ = LagLlamaEstimator(
            ckpt_path="lag-llama.ckpt",
            prediction_length=self.prediction_length_,
            context_length=self.context_length_,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            batch_size=self.batch_size_,
            device=self.device_,
            rope_scaling=rope_scaling_arguments,
        )

        lightning_module = self.estimator_.create_lightning_module()
        transformation = self.estimator_.create_transformation()

        # Finally, we create our predictor!
        self.predictor_ = self.estimator_.create_predictor(
            transformation, lightning_module
        )

        # Since we are importing pretrained weights
        self._is_fitted = True

    # todo: implement this, mandatory
    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : GluonTS PandasDataset Object, optional (default=None)
            Time series to which to fit the forecaster.

        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.

        X : GluonTS PandasDataset Object, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        from lag_llama.gluon.estimator import LagLlamaEstimator

        # Creating a new LagLlama estimator with the appropriate
        # forecasting horizon
        self.estimator_ = LagLlamaEstimator(
            ckpt_path="lag-llama.ckpt",
            prediction_length=32 if not fh else fh,  # This is the most important here!
            context_length=self.context_length_,
            input_size=self.estimator_args["input_size"],
            n_layer=self.estimator_args["n_layer"],
            n_embd_per_head=self.estimator_args["n_embd_per_head"],
            n_head=self.estimator_args["n_head"],
            scaling=self.estimator_args["scaling"],
            time_feat=self.estimator_args["time_feat"],
            batch_size=self.batch_size_,
            device=self.device_,
            lr=self.lr_,
            trainer_kwargs=self.trainer_kwargs_,
        )

        lightning_module = self.estimator.create_lightning_module()
        transformation = self.estimator.create_transformation()

        # Creating a new predictor
        self.predictor_ = self.estimator_.create_predictor(
            transformation, lightning_module
        )

        # Lastly, training the model
        self.predictor_ = self.estimator.train(y.train, cache_data=True)

    def _predict(self, fh=None, X=None):
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
            The forecasting horizon is not read here, please set it as
            `prediction_length` during initialization or when calling the fit function

        X : GluonTS PandasDataset Object (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : GluonTS PandasDataset Object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        from gluonts.evaluation import make_evaluation_predictions

        # Creating a forecaster object
        forecast_it, _ = make_evaluation_predictions(
            dataset=X, predictor=self.predictor_, num_samples=self.num_samples_
        )

        # Forming a list of the forecasting iterations
        forecasts = list(forecast_it)

        return forecasts

    def _predict_interval(self, fh, X, coverage):
        """Compute/return prediction quantiles for a forecast.

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
        """
        import numpy as np
        import pandas as pd
        from gluonts.evaluation import make_evaluation_predictions

        # Obtaining our evaluations
        forecast_it, _ = make_evaluation_predictions(
            dataset=X, predictor=self.predictor_, num_samples=self.num_samples_
        )

        forecasts = list(forecast_it)

        # Stores all intervals
        intervals = []

        for forecast in forecasts:
            samples = forecast.samples

            # Creating a DataFrame for this forecast
            df = pd.DataFrame(index=forecast.index)

            for c in coverage:
                # Defining lower and upper interval values
                df[("target", c, "lower")] = np.min(samples)
                df[("target", c, "upper")] = np.max(samples)

            intervals.append(df)

        pred_int = pd.concat(intervals, axis=0)

        return pred_int

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
        params = [
            {
                "device": "cpu",
                "context_length": 32,
                "prediction_length": 100,
                "num_samples": 10,
                "batch_size": 32,
                "nonnegative_pred_samples": False,
                "lr": 5e-5,
                "trainer_kwargs": {"num_epochs": 10},
            }
        ]

        return params
