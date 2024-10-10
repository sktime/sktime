# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements an adapter for the LagLlama estimator for intergration into sktime."""

__author__ = ["shlok191"]


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
        "y_inner_mtype": ["gluonts_ListDataset_panel", "gluonts_ListDataset_series"],
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

    def _reform_y(self, y):
        from gluonts.dataset.common import ListDataset

        if y is None:
            return None

        shape = y[0]["target"].shape

        if len(shape) == 2 and shape[1] == 1:
            new_values = []

            # Updating the ListDataset to flatten univariate target values
            for data_entry in y:
                target = data_entry["target"]

                if len(target.shape) == 2 and target.shape[1] == 1:
                    data_entry["target"] = target.flatten()

                new_values.append(data_entry)

            new_y = ListDataset(new_values, one_dim_target=True, freq="D")

            return new_y

        return y

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
            prediction_length=len(fh),
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
        self.predictor_ = self.estimator_.create_predictor(
            transformation, lightning_module
        )

        # Updating y value to make it compatible with LagLlama
        y = self._reform_y(y)

        # Lastly, training the model
        self.predictor_ = self.estimator_.train(
            y, cache_data=True, shuffle_buffer_length=self.shuffle_buffer_length_
        )

    def convert_sampleforecast_to_listdataset(forecasts):
        """Convert a GluonTS SampleForecast to a ListDataset.

        Parameters
        ----------
        forecast : GluonTS SampleForecast
            The SampleForecast generated by LagLlama

        Returns
        -------
        GluonTS ListDataset
            Returns a ListDatset containing len(time_series) * num_samples entries
        """
        from gluonts.dataset.common import ListDataset
        from gluonts.model.forecast import SampleForecast

        data_entries = []

        for index, forecast in enumerate(forecasts):
            # Assert that each entry is a valid SampleForecast
            assert isinstance(
                forecast, SampleForecast
            ), f"forecast_{index} is not a valid SampleForecast."

            # Obtaining individual elements that build a ListDataset
            item_id = forecast.item_id
            start_date = forecast.start_date.to_timestamp()
            samples = forecast.samples

            # Formulating our data
            data = [
                {"item_id": item_id, "start": start_date, "target": sample}
                for sample in samples
            ]

            data_entries.extend(data)

        return ListDataset(data_entries, freq=forecasts[0].start_date.freq)

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

        # Creating a forecaster object
        forecasts, _ = make_evaluation_predictions(
            dataset=y, predictor=self.predictor_, num_samples=self.num_samples_
        )

        # Forming a list of the forecasting iterations
        forecasts = list(forecasts)
        list_dataset = self.convert_sampleforecast_to_listdataset(forecasts)

        return list_dataset

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
