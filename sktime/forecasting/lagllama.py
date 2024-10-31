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
            if None in y.index.names:
                y.index.names = ["item_id", "timepoints"]
            item_id = y.index.names[0]
            timepoint = y.index.names[-1]

            self._df_config = {
                "target": [target_col],
                "item_id": item_id,
                "timepoints": timepoint,
            }

            # Reset the index to make it compatible with GluonTS
            y = y.reset_index()
            y.set_index(timepoint, inplace=True)

            dataset = PandasDataset.from_long_dataframe(
                y, target=target_col, item_id=item_id, future_length=0
            )

        else:
            self._df_config = {
                "target": [target_col],
            }
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

    def infer_freq(self, index):
        """
        Infer frequency of the index.

        Parameters
        ----------
        index: pd.Index
            Index of the time series data.

        Notes
        -----
        Uses only first 3 values of the index to infer the frequency.
        As `freq=None` is returned in case of multiindex timepoints.

        """
        if isinstance(index, pd.PeriodIndex):
            return index.freq
        return pd.infer_freq(index[:3])

    def _extend_df(self, df, fh):
        """Extend the input dataframe up to the timepoints that need to be predicted.

        Parameters
        ----------
        df : pd.DataFrame
            Input data that needs to be extended
        X : pd.DataFrame, default=None
            Assumes that X has future timepoints and is
            concatenated to the input data,
            if X is present in the input, but None here the values of X are assumed
            to be 0 in future timepoints that need to be predicted.
        is_range_index : bool, default=False
            If True, the index is a range index.
        is_period_index : bool, default=False
            If True, the index is a period index.

        Returns
        -------
        pd.DataFrame
            Extended dataframe with future timepoints.
        """
        index = self.return_time_index(df)
        # Extend the index to the future timepoints
        # respective to index last seen

        if self.check_range_index(df):
            pred_index = pd.RangeIndex(
                self.cutoff[0] + 1, self.cutoff[0] + max(self.fh._values) + 1
            )
        elif isinstance(index, pd.PeriodIndex):
            pred_index = pd.period_range(
                self.cutoff[0],
                periods=max(self.fh._values) + 1,
                freq=index.freq,
            )[1:]
        else:
            pred_index = pd.date_range(
                self.cutoff[0],
                periods=max(self.fh._values) + 1,
                freq=self.infer_freq(index),
            )[1:]

        if isinstance(df.index, pd.MultiIndex):
            # Works for any number of levels in the MultiIndex
            index_levels = [
                df.index.get_level_values(i).unique()
                for i in range(df.index.nlevels - 1)
            ]
            index_levels.append(pred_index)
            new_index = pd.MultiIndex.from_product(index_levels, names=df.index.names)
        else:
            new_index = pred_index

        df_y = pd.DataFrame(columns=df.columns, index=new_index)
        df_y.fillna(0, inplace=True)
        extended_df = pd.concat([df, df_y])
        extended_df = extended_df.sort_index()
        return extended_df

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

        if y is None:
            y = self._y
            _y = self._y.copy()
        else:
            _y = y.copy()

        _y = self._extend_df(_y, fh)

        self._is_range_index = False
        if self.check_range_index(y):
            _y.index = self.handle_range_index(_y.index)
            self._is_range_index = True

        _y = self._convert_to_float(_y)
        dataset = self._get_gluonts_dataset(_y)

        # Forming a list of the forecasting iterations
        forecast_it, _ = make_evaluation_predictions(
            dataset=dataset, predictor=self.model, num_samples=100
        )
        predictions = self._get_prediction_df(forecast_it, self._df_config)

        if self._is_range_index:
            timepoints = self.return_time_index(predictions)
            timepoints = timepoints.to_timestamp()
            timepoints = (timepoints - pd.Timestamp("2010-01-01")).map(
                lambda x: x.days
            ) + self.return_time_index(y)[0]
            if isinstance(predictions.index, pd.MultiIndex):
                predictions.index = predictions.index.set_levels(
                    levels=timepoints.unique(), level=-1
                )
                # Convert str type to int
                predictions.index = predictions.index.map(lambda x: (int(x[0]), x[1]))
            else:
                predictions.index = timepoints

        return predictions

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

    def _get_prediction_df(self, forecast_iter, df_config):
        def handle_series_prediction(forecast, target):
            # Renames the predicted column to the target column name
            pred = forecast.mean_ts
            if target[0] is not None:
                return pred.rename(target[0])
            else:
                return pred

        def handle_panel_predictions(forecasts_it, df_config):
            # Convert all panel forecasts to a single panel dataframe
            panels = []
            for forecast in forecasts_it:
                df = forecast.mean_ts.reset_index()
                df.columns = [df_config["timepoints"], df_config["target"][0]]
                df[df_config["item_id"]] = forecast.item_id
                df.set_index(
                    [df_config["item_id"], df_config["timepoints"]], inplace=True
                )
                panels.append(df)
            return pd.concat(panels)

        forecasts = list(forecast_iter)

        # Assuming all forecasts_it are either series or panel type.
        if forecasts[0].item_id is None:
            return handle_series_prediction(forecasts[0], df_config["target"])
        else:
            return handle_panel_predictions(forecasts, df_config)

    def return_time_index(self, df):
        """Return the time index, given any type of index."""
        if isinstance(df.index, pd.MultiIndex):
            return df.index.get_level_values(-1)
        else:
            return df.index

    def check_range_index(self, df):
        """Check if the index is a range index."""
        timepoints = self.return_time_index(df)
        if isinstance(timepoints, pd.RangeIndex):
            return True
        elif pd.api.types.is_integer_dtype(timepoints):
            return True
        return False

    def handle_range_index(self, index):
        """
        Convert RangeIndex to Dummy DatetimeIndex.

        As gluonts PandasDataset expects a DatetimeIndex.
        """
        start_date = "2010-01-01"
        if isinstance(index, pd.MultiIndex):
            n_periods = index.get_level_values(1).nunique()
            panels = index.get_level_values(0).unique()
            datetime_index = pd.date_range(
                start=start_date, periods=n_periods, freq="D"
            )
            new_index = pd.MultiIndex.from_product([panels, datetime_index])
        else:
            n_periods = index.size
            new_index = pd.date_range(start=start_date, periods=n_periods, freq="D")
        return new_index

    def _convert_hierarchical_to_panel(self, df):
        # Flatten the MultiIndex to a panel type DataFrame
        data = df.copy()
        flattened_index = [("*".join(map(str, x[:-1])), x[-1]) for x in data.index]
        # Create a new MultiIndex with the flattened level and the last level unchanged
        data.index = pd.MultiIndex.from_tuples(
            flattened_index, names=["Flattened_Level", data.index.names[-1]]
        )
        return data

    def _convert_panel_to_hierarchical(self, df, original_index_names=None):
        # Store the original index names
        if original_index_names is None:
            original_index_names = df.index.names

        # Reset the index to get 'Flattened_Level' as a column
        data = df.reset_index()

        # Split the 'Flattened_Level' column into multiple columns
        split_levels = data["Flattened_Level"].str.split("*", expand=True)
        split_levels.columns = original_index_names[:-1]
        # Get the names of the split levels as a list of column names
        index_names = split_levels.columns.tolist()

        # Combine the split levels with the rest of the data
        data_converted = pd.concat(
            [split_levels, data.drop(columns=["Flattened_Level"])], axis=1
        )

        # Get the last index name if it exists, otherwise use a default name
        last_index_name = (
            original_index_names[-1]
            if original_index_names[-1] is not None
            else "timepoints"
        )

        # Set the new index with the split levels and the last index name
        data_converted = data_converted.set_index(index_names + [last_index_name])

        return data_converted
