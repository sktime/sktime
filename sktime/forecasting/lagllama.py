# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements an adapter for the LagLlama estimator for intergration into sktime."""

__author__ = ["shlok191"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _BaseGlobalForecaster


class LagLlamaForecaster(_BaseGlobalForecaster):
    """Base class that interfaces the LagLlama forecaster.

    Parameters
    ----------
    weights_url : str, optional (default="time-series-foundation-models/Lag-Llama")'
        The URL of the weights on hugging face for the LagLlama estimator to fetch.

    device : str, optional (default="cpu")
        Specifies the device on which to load the model.

    context_length: int, optional (default=32)
        The number of prior timestep data entries provided.

    num_samples: int, optional (default=10)
        Number of sample paths desired for evaluation.

    batch_size: int, optional (default=32)
        The number of batches to train for in parallel.

    nonnegative_pred_samples: bool, optional (default=False)
        If True, ensures all predicted samples are passed
        through ReLU,and are thus positive or 0.

    lr: float, optional (default=5e-5)
        The learning rate of the model.

    shuffle_buffer_length: int, optional (default=1000)
        The size of the buffer from which training samples are drawn

    trainer_kwargs: dict, optional (default={"num_epochs": 50})
        The arguments to pass to the GluonTS trainer.

    Examples
    --------
    >>> from gluonts.dataset.repository.datasets import get_dataset
    >>> from sktime.forecasting.lagllama import LagLlamaForecaster
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from gluonts.dataset.common import ListDataset

    >>> dataset = get_dataset("m4_weekly")

    # Converts to a GluonTS ListDataset, a format supported by sktime!
    >>> train_dataset = ListDataset(dataset.train, freq='W')
    >>> test_dataset = ListDataset(dataset.test, freq='W')

    >>> forecaster = LagLlamaForecaster(
    ...     context_length=dataset.metadata.prediction_length * 3,
    ...     lr=5e-4,
    ...     )

    >>> fh=ForecastingHorizon(range(dataset.metadata.prediction_length))
    >>> forecaster.fit(y=train_dataset,fh = fh)
    >>> y_pred = forecaster.predict(y=test_dataset)

    >>> y_pred
    """

    _tags = {
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["gluonts_ListDataset_panel", "gluonts_ListDataset_series"],
        "scitype:y": "both",
        "ignores-exogeneous-X": True,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:insample": True,
        "capability:global_forecasting": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "authors": ["shlok191"],
        "maintainers": ["shlok191"],
        "python_version": None,
        "python_dependencies": ["gluonts", "huggingface-hub"],
    }

    def __init__(
        self,
        train=False,
        model_path=None,
        device=None,
        context_length=None,
        num_samples=None,
        batch_size=None,
        nonnegative_pred_samples=None,
        lr=None,
        trainer_kwargs=None,
        shuffle_buffer_length=None,
        use_source_package=False,
    ):
        # Initializing parent class
        super().__init__()

        import torch
        from huggingface_hub import hf_hub_download

        # Defining private variable values
        self.model_path = model_path
        self.model_path_ = (
            "time-series-foundation-models/Lag-Llama" if not model_path else model_path
        )

        self.device = device
        self.device_ = torch.device("cpu") if not device else torch.device(device)

        self.context_length = context_length
        self.context_length_ = 32 if not context_length else context_length

        self.num_samples = num_samples
        self.num_samples_ = 10 if not num_samples else num_samples

        self.batch_size = batch_size
        self.batch_size_ = 32 if not batch_size else batch_size

        # Now storing the training related variables
        self.lr = lr
        self.lr_ = 5e-5 if not lr else lr

        self.shuffle_buffer_length = shuffle_buffer_length
        self.shuffle_buffer_length_ = (
            1000 if not shuffle_buffer_length else shuffle_buffer_length
        )

        self.train = train
        self.trainer_kwargs = trainer_kwargs
        self.trainer_kwargs_ = (
            {"max_epochs": 10} if not trainer_kwargs else trainer_kwargs
        )

        # Not storing private variables for boolean specific values
        self.nonnegative_pred_samples = nonnegative_pred_samples

        # Downloading the LagLlama weights from Hugging Face
        self.ckpt_url_ = hf_hub_download(
            repo_id=self.model_path_, filename="lag-llama.ckpt"
        )

        # Load in the lag llama checkpoint
        ckpt = torch.load(self.ckpt_url_, map_location=self.device_)

        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]
        self.estimator_args = estimator_args

        self.use_source_package = use_source_package

    def _get_gluonts_dataset(self, y):
        from gluonts.dataset.pandas import PandasDataset

        target_col = str(y.columns[0])
        if isinstance(y.index, pd.MultiIndex):
            item_id = y.index.names[0]
            timepoint = y.index.names[-1]

            # Reset the index to make it compatible with GluonTS
            y = y.reset_index()
            y.set_index(timepoint, inplace=True)

            dataset = PandasDataset.from_long_dataframe(
                y, target=target_col, item_id=item_id, future_length=0
            )

        else:
            dataset = PandasDataset(y, future_length=0, target=target_col)

        return dataset

    def _convert_to_float(self, df):
        for col in df.columns:
            # Check if column is not of string type
            if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype("float32")

        return df

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : GluonTS ListDataset Object, optional (default=None)
            Time series to which to fit the forecaster.

        fh : guaranteed to be ForecastingHorizon or None
            The length of future of timesteps to predict

        X : GluonTS ListDataset Object, optional (default=None)
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        if self.use_source_package:
            if _check_soft_dependencies("lag-llama"):
                from lag_llama.gluon.estimator import LagLlamaEstimator
        else:
            from sktime.libs.lag_llama.gluon.estimator import LagLlamaEstimator

        # Creating a new LagLlama estimator with the appropriate
        # forecasting horizon
        self.estimator_ = LagLlamaEstimator(
            ckpt_path=self.ckpt_url_,
            prediction_length=max(fh.to_relative(self.cutoff)),
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

        lightning_module = self.estimator_.create_lightning_module()
        transformation = self.estimator_.create_transformation()

        # Creating a new predictor
        self.model = self.estimator_.create_predictor(transformation, lightning_module)

        if self.train:
            # Updating y value to make it compatible with LagLlama
            y = self._convert_to_float(y)
            y = self._get_gluonts_dataset(y)
            # Lastly, training the model
            self.model = self.estimator_.train(
                y, cache_data=True, shuffle_buffer_length=self.shuffle_buffer_length_
            )

    def _extend_df(self, df, fh):
        prediction_length = max(fh.to_relative(self.cutoff))
        freq = df.index.freq
        extend_df = pd.period_range(
            start=df.index[-1], periods=prediction_length + 1, freq=freq
        )[1:]

        return pd.concat([df, pd.DataFrame(index=extend_df)], axis=0)

    def _predict(self, fh, X=None, y=None):
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

        X : GluonTS ListDataset Object (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        y : GluonTS ListDataset Object (default=None)
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")

        Returns
        -------
        y_pred : GluonTS ListDataset Object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        from gluonts.evaluation import make_evaluation_predictions

        if y is not None:
            y = pd.concat([self._y, y], axis=0)
        else:
            y = self._extend_df(self._y, fh)

        y = self._convert_to_float(y)
        dataset = self._get_gluonts_dataset(y)

        # Forming a list of the forecasting iterations
        forecast_it, _ = make_evaluation_predictions(
            dataset=dataset, predictor=self.model, num_samples=100
        )
        forecasts = list(forecast_it)
        return forecasts[0].mean_ts

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
        forecasts, _ = make_evaluation_predictions(
            dataset=X, predictor=self.predictor_, num_samples=self.num_samples_
        )

        forecasts = list(forecasts)

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
                "context_length": 50,
                "num_samples": 16,
                "batch_size": 32,
                "shuffle_buffer_length": 64,
                "lr": 5e-5,
            },
            {
                "context_length": 50,
                "num_samples": 16,
                "batch_size": 32,
                "shuffle_buffer_length": 64,
                "lr": 5e-5,
            },
        ]

        return params
