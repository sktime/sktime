# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements an adapter for the LagLlama estimator for intergration into sktime."""

__author__ = ["shlok191"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _BaseGlobalForecaster


class LagLlamaForecaster(_BaseGlobalForecaster):
    """LagLlama Foundation Model for Zero-Shot Time Series Forecasting.

    LagLlama is a foundation model for univariate probabilistic time series forecasting
    based on a decoder-only transformer architecture. This implementation provides
    zero-shot prediction using pretrained weights from HuggingFace.

    The model checkpoint is automatically downloaded on first use if not provided.

    Parameters
    ----------
    ckpt_path : str, optional (default=None)
        Path to LagLlama checkpoint file. If None, automatically downloads
        from HuggingFace: "time-series-foundation-models/Lag-Llama".
    device : str, optional (default=None)
        Device for inference ("cpu", "cuda", "cuda:0", etc.).
        If None, uses CUDA if available, otherwise CPU.
    context_length : int, optional (default=32)
        Number of past time steps used as context for prediction.
        LagLlama was trained with context_length=32.
    num_samples : int, optional (default=100)
        Number of sample paths for probabilistic forecasting.
    batch_size : int, optional (default=1)
        Batch size for prediction.
    use_rope_scaling : bool, optional (default=False)
        Whether to use RoPE scaling for handling longer context lengths.
    nonnegative_pred_samples : bool, optional (default=False)
        If True, ensures all predicted samples are passed through ReLU.
    use_source_package : bool, optional (default=False)
        If True, uses the external lag-llama package instead of vendored version.

    Examples
    --------
    >>> from sktime.forecasting.lagllama import LagLlamaForecaster  # doctest: +SKIP
    >>> from sktime.forecasting.base import ForecastingHorizon  # doctest: +SKIP
    >>> from sktime.datasets import load_airline  # doctest: +SKIP
    >>>
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = LagLlamaForecaster(  # doctest: +SKIP
    ...     context_length=32,
    ...     num_samples=100
    ... )
    >>> fh = ForecastingHorizon([1, 2, 3, 4, 5, 6])  # doctest: +SKIP
    >>> forecaster.fit(y, fh=fh)  # doctest: +SKIP
    LagLlamaForecaster(...)
    >>> y_pred = forecaster.predict()  # Point predictions  # doctest: +SKIP
    >>> # 90% prediction intervals
    >>> y_interval = forecaster.predict_interval(coverage=0.9)  # doctest: +SKIP

    References
    ----------
    .. [1] Rasul, Kashif, et al. "Lag-Llama: Towards Foundation Models for
           Probabilistic Time Series Forecasting."
           arXiv preprint arXiv:2310.08278 (2023).
    """

    _tags = {
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "X_inner_mtype": ["pd.DataFrame"],
        "scitype:y": "both",
        "capability:exogenous": False,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": True,
        "capability:global_forecasting": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": True,
        "authors": ["shlok191"],
        "maintainers": ["shlok191"],
        "python_version": None,
        "python_dependencies": [
            "gluonts>=0.14.0",
            "torch",
            "lightning>=2.0,<2.6",
            "huggingface-hub",
        ],
    }

    def __init__(
        self,
        ckpt_path=None,
        device=None,
        context_length=32,
        num_samples=100,
        batch_size=1,
        use_rope_scaling=False,
        nonnegative_pred_samples=False,
        use_source_package=False,
    ):
        """Initialize LagLlamaForecaster.

        Parameters
        ----------
        ckpt_path : str, optional (default=None)
            Path to LagLlama checkpoint file. If None, automatically downloads
            from HuggingFace: "time-series-foundation-models/Lag-Llama"
        device : str, optional (default=None)
            Device for inference ("cpu", "cuda", "cuda:0", etc.).
            If None, uses CUDA if available, otherwise CPU.
        context_length : int, optional (default=32)
            Number of past time steps used as context for prediction.
            LagLlama was trained with context_length=32.
        num_samples : int, optional (default=100)
            Number of sample paths for probabilistic forecasting.
        batch_size : int, optional (default=1)
            Batch size for prediction.
        use_rope_scaling : bool, optional (default=False)
            Whether to use RoPE scaling for handling longer context lengths.
        nonnegative_pred_samples : bool, optional (default=False)
            If True, ensures all predicted samples are passed through ReLU.
        use_source_package : bool, optional (default=False)
            If True, uses the external lag-llama package instead of vendored version.
        """
        # Initialize parent class
        super().__init__()

        import torch

        # Store parameters
        self.ckpt_path = ckpt_path
        self.device = device
        self.context_length = context_length
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.use_rope_scaling = use_rope_scaling
        self.nonnegative_pred_samples = nonnegative_pred_samples
        self.use_source_package = use_source_package

        # Set device (lazy - actual device object created when needed)
        if device is None:
            self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device_ = torch.device(device)

    def _ensure_checkpoint(self):
        """Download checkpoint from HuggingFace if not found locally.

        Returns
        -------
        str
            Path to the checkpoint file.
        """
        import os

        from huggingface_hub import hf_hub_download

        # If a local path is provided and exists, use it
        if self.ckpt_path and os.path.exists(self.ckpt_path):
            return self.ckpt_path

        # Otherwise, download from HuggingFace (uses default cache directory)
        ckpt_path = hf_hub_download(
            repo_id="time-series-foundation-models/Lag-Llama",
            filename="lag-llama.ckpt",
        )

        return ckpt_path

    def _get_gluonts_dataset(self, y):
        from gluonts.dataset.pandas import PandasDataset

        target_col = y.columns[0]
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

        For LagLlama, this creates the zero-shot predictor from pretrained weights.
        No training/fine-tuning is performed - only model initialization.

        Parameters
        ----------
        y : pd.DataFrame
            Time series to which to fit the forecaster.
            Guaranteed to be of mtype in self.get_tag("y_inner_mtype").
        X : pd.DataFrame, optional (default=None)
            Exogeneous time series (ignored by LagLlama).
        fh : ForecastingHorizon
            The forecasting horizon.

        Returns
        -------
        self : reference to self
        """
        import torch

        # Import LagLlama estimator
        if self.use_source_package:
            if _check_soft_dependencies("lag-llama", severity="warning"):
                from lag_llama.gluon.estimator import LagLlamaEstimator
            else:
                from sktime.libs.lag_llama.gluon.estimator import LagLlamaEstimator
        else:
            from sktime.libs.lag_llama.gluon.estimator import LagLlamaEstimator

        # Get or download checkpoint
        ckpt_path = self._ensure_checkpoint()

        # Load checkpoint with PyTorch 2.6+ compatibility
        ckpt = torch.load(ckpt_path, map_location=self.device_, weights_only=False)
        estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

        # Setup RoPE scaling if requested
        rope_scaling_arguments = None
        if self.use_rope_scaling:
            prediction_length = max(fh.to_relative(self.cutoff))
            rope_scaling_arguments = {
                "type": "linear",
                "factor": max(
                    1.0,
                    (self.context_length + prediction_length)
                    / estimator_args["context_length"],
                ),
            }

        # Create LagLlama estimator
        self.estimator_ = LagLlamaEstimator(
            ckpt_path=ckpt_path,
            prediction_length=max(fh.to_relative(self.cutoff)),
            context_length=self.context_length,
            input_size=estimator_args["input_size"],
            n_layer=estimator_args["n_layer"],
            n_embd_per_head=estimator_args["n_embd_per_head"],
            n_head=estimator_args["n_head"],
            scaling=estimator_args["scaling"],
            time_feat=estimator_args["time_feat"],
            rope_scaling=rope_scaling_arguments,
            batch_size=self.batch_size,
            num_parallel_samples=self.num_samples,
            device=self.device_,
        )

        # Create predictor with PyTorch 2.6+ compatibility patch
        # Lightning uses weights_only=True by default which causes issues
        original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            lightning_module = self.estimator_.create_lightning_module()
            transformation = self.estimator_.create_transformation()
            self.predictor_ = self.estimator_.create_predictor(
                transformation, lightning_module
            )
        finally:
            # Restore original torch.load
            torch.load = original_load

        return self

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
            The forecasting horizon with the steps ahead to to predict.
            If not passed in _fit, guaranteed to be passed here
        X : optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.

        Returns
        -------
        y_pred : pd.DataFrame
            Point predictions with same index type as input y
        """
        from gluonts.evaluation import make_evaluation_predictions

        if y is None:
            y = self._y
            _y = self._y.copy()
        else:
            _y = y.copy()

        _y = self._extend_df(_y, fh)

        # Check for range index
        self._is_range_index = False
        if self.check_range_index(y):
            _y.index = self.handle_range_index(_y.index)
            self._is_range_index = True

        # Check for hierarchical data and convert to panel
        _is_hierarchical = False
        _original_index_names = None
        if _y.index.nlevels >= 3:
            _original_index_names = _y.index.names
            _y = self._convert_hierarchical_to_panel(_y)
            _is_hierarchical = True

        _y = self._convert_to_float(_y)
        dataset = self._get_gluonts_dataset(_y)

        # Forming a list of the forecasting iterations
        forecast_it, _ = make_evaluation_predictions(
            dataset=dataset, predictor=self.predictor_, num_samples=self.num_samples
        )
        predictions = self._get_prediction_df(forecast_it, self._df_config)

        # Convert back to hierarchical if needed
        if _is_hierarchical:
            predictions = self._convert_panel_to_hierarchical(
                predictions, _original_index_names
            )

        # Handle range index conversion back
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

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Compute quantile forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (ignored).
        alpha : list of float, optional (default=None)
            The quantiles to predict. If None, uses default [0.1, 0.25, 0.5, 0.75, 0.9].

        Returns
        -------
        quantiles : pd.DataFrame
            Quantile forecasts with MultiIndex (alpha, time) or
            MultiIndex (alpha, item, time) for panel data.
        """
        from gluonts.evaluation import make_evaluation_predictions

        if alpha is None:
            alpha = [0.1, 0.25, 0.5, 0.75, 0.9]

        # Get the data (same as _predict)
        y = self._y
        _y = self._y.copy()

        _y = self._extend_df(_y, fh)

        # Handle range index
        original_is_range_index = False
        if self.check_range_index(y):
            _y.index = self.handle_range_index(_y.index)
            original_is_range_index = True

        # Check for hierarchical data and convert to panel
        _is_hierarchical = False
        _original_index_names = None
        if _y.index.nlevels >= 3:
            _original_index_names = _y.index.names
            _y = self._convert_hierarchical_to_panel(_y)
            _is_hierarchical = True

        _y = self._convert_to_float(_y)
        dataset = self._get_gluonts_dataset(_y)

        # Get predictions with samples
        forecast_it, _ = make_evaluation_predictions(
            dataset=dataset, predictor=self.predictor_, num_samples=self.num_samples
        )

        forecasts = list(forecast_it)

        # Extract quantiles for each forecast
        quantile_dfs = []

        for forecast in forecasts:
            # GluonTS forecasts have .quantile(q) method
            forecast_quantiles = {}
            for q in alpha:
                forecast_quantiles[q] = forecast.quantile(q)

            # Build DataFrame for this forecast
            if forecast.item_id is not None:
                # Panel data - need MultiIndex (alpha, item_id, timepoints)
                for q in alpha:
                    q_series = forecast_quantiles[q]
                    if isinstance(q_series, pd.Series):
                        df = q_series.reset_index()
                        df.columns = [self._df_config["timepoints"], "quantile"]
                        df["alpha"] = q
                        df[self._df_config["item_id"]] = forecast.item_id
                        quantile_dfs.append(df)
            else:
                # Single series - MultiIndex (alpha, timepoints)
                for q in alpha:
                    q_series = forecast_quantiles[q]
                    if isinstance(q_series, pd.Series):
                        df = q_series.to_frame(name="quantile")
                        df["alpha"] = q
                        df = df.reset_index()
                        df.columns = ["timepoints", "quantile", "alpha"]
                        quantile_dfs.append(df)

        # Combine all quantile forecasts
        if len(quantile_dfs) > 0:
            result = pd.concat(quantile_dfs, ignore_index=True)

            # Set appropriate index
            if forecasts[0].item_id is not None:
                # Panel: MultiIndex (alpha, item_id, timepoints)
                result = result.set_index(
                    ["alpha", self._df_config["item_id"], self._df_config["timepoints"]]
                )
                result = result["quantile"].unstack(level=0).T
            else:
                # Single series: MultiIndex (alpha, timepoints)
                result = result.set_index(["alpha", "timepoints"])
                result = result["quantile"].unstack(level=0).T

            # Convert back to hierarchical if needed
            if _is_hierarchical:
                result = self._convert_panel_to_hierarchical(
                    result, _original_index_names
                )

            # Handle range index conversion back
            if original_is_range_index:
                timepoints = self.return_time_index(result)
                timepoints = timepoints.to_timestamp()
                timepoints = (timepoints - pd.Timestamp("2010-01-01")).map(
                    lambda x: x.days
                ) + self.return_time_index(y)[0]

                if isinstance(result.index, pd.MultiIndex):
                    result.index = result.index.set_levels(
                        levels=timepoints.unique(), level=-1
                    )
                else:
                    result.index = timepoints

            return result

        return pd.DataFrame()

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
                "context_length": 32,
                "num_samples": 10,  # Reduced for faster tests
                "batch_size": 1,
            },
            {
                "context_length": 64,
                "num_samples": 20,
                "use_rope_scaling": True,
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
