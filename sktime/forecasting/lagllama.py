# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements an adapter for the LagLlama estimator for intergration into sktime."""

__author__ = ["shlok191", "pranavvp16"]

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _BaseGlobalForecaster


class LagLlamaForecaster(_BaseGlobalForecaster):
    """LagLlama Foundation Model for Time Series Forecasting.

    LagLlama is a foundation model for univariate probabilistic time series forecasting
    based on a decoder-only transformer architecture. This implementation supports
    both zero-shot prediction using pretrained weights and fine-tuning on custom data.

    The model checkpoint is automatically downloaded on first use if not provided.

    **Fit Strategies: Zero-Shot and Fine-Tuning**

    This model supports two fit strategies via the ``fit_strategy`` parameter:

    - **zero-shot** (default): Uses pretrained model as-is without training.
      Fast inference with no training overhead. Suitable for quick predictions.

    - **finetuning**: Fine-tunes all model parameters on the provided data.
      Takes longer but may improve accuracy on specific datasets.
      Controlled by ``trainer_kwargs``, ``lr``, and ``aug_prob`` parameters.
      Supports validation split for all data types including hierarchical data.

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
    **Zero-shot forecasting (default)**

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

    **Fine-tuning on custom data**

    >>> from sktime.forecasting.lagllama import LagLlamaForecaster  # doctest: +SKIP
    >>> from sktime.datasets import load_airline  # doctest: +SKIP
    >>>
    >>> y = load_airline()  # doctest: +SKIP
    >>> forecaster = LagLlamaForecaster(  # doctest: +SKIP
    ...     fit_strategy="finetuning",
    ...     context_length=32,
    ...     num_samples=100,
    ...     trainer_kwargs={"max_epochs": 10},
    ...     lr=5e-4,
    ...     validation_split=0.2
    ... )
    >>> forecaster.fit(y, fh=[1, 2, 3, 4, 5, 6])  # doctest: +SKIP
    LagLlamaForecaster(...)
    >>> y_pred = forecaster.predict()  # doctest: +SKIP

    References
    ----------
    .. [1] Rasul, Kashif, et al. "Lag-Llama: Towards Foundation Models for
           Probabilistic Time Series Forecasting."
           arXiv preprint arXiv:2310.08278 (2023).
    """

    _tags = {
        "y_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "univariate",  # LagLlama is univariate only
        "capability:exogenous": False,
        "capability:multivariate": False,  # LagLlama is univariate only
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:global_forecasting": True,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "authors": ["shlok191"],
        "maintainers": ["shlok191"],
        "python_version": "<3.14",
        "python_dependencies": [
            "gluonts>=0.14.0",
            "torch",
            "lightning>=2.0,<2.6",
            "huggingface-hub",
        ],
        "tests:vm": True,
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
        fit_strategy="zero-shot",
        validation_split=0.2,
        trainer_kwargs=None,
        lr=5e-4,
        aug_prob=0.0,
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
        fit_strategy : str, optional (default="zero-shot")
            Strategy to use for fitting the model. One of:
            - "zero-shot": Use pretrained model as-is without training
            - "finetuning": Fine-tune all model parameters on the provided data
        validation_split : float, optional (default=0.2)
            Fraction of data to use for validation during fine-tuning.
            Only used when fit_strategy="finetuning". Set to None to skip validation.
        trainer_kwargs : dict, optional (default=None)
            Additional arguments for PyTorch Lightning Trainer during fine-tuning.
            Only used when fit_strategy="finetuning".
            If None, defaults to {"max_epochs": 50}.
            Common options: "max_epochs", "devices", "accelerator", etc.
        lr : float, optional (default=5e-4)
            Learning rate for fine-tuning.
            Only used when fit_strategy="finetuning".
        aug_prob : float, optional (default=0.0)
            Probability of applying data augmentation during training.
            Only used when fit_strategy="finetuning".
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
        self.fit_strategy = fit_strategy
        self.validation_split = validation_split
        self.trainer_kwargs = trainer_kwargs
        self._trainer_kwargs = (
            trainer_kwargs if trainer_kwargs is not None else {"max_epochs": 50}
        )
        self.lr = lr
        self.aug_prob = aug_prob

        # Validate fit_strategy
        if self.fit_strategy not in ["zero-shot", "finetuning"]:
            raise ValueError(
                f"fit_strategy must be 'zero-shot' or 'finetuning', "
                f"got '{self.fit_strategy}'"
            )

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
            # Check if hierarchical (3+ levels) and convert to panel format
            # GluonTS requires panel (2-level) data, so flatten hierarchical data
            if y.index.nlevels >= 3:
                y = self._convert_hierarchical_to_panel(y.copy())

            if None in y.index.names:
                y.index.names = ["item_id", "timepoints"]
            item_id = y.index.names[0]
            timepoint = y.index.names[-1]

            self._df_config = {
                "target": [target_col],
                "item_id": item_id,
                "timepoints": timepoint,
            }

            # Infer frequency from the original index before resetting
            # This is needed for hierarchical data where GluonTS cannot infer
            # it after reset
            time_index = y.index.get_level_values(-1)
            freq = self.infer_freq(time_index)

            # Reset the index to make it compatible with GluonTS
            y = y.reset_index()
            y.set_index(timepoint, inplace=True)

            # Pass frequency explicitly to avoid inference errors with hierarchical data
            # GluonTS freq inference can fail when data has been reset from MultiIndex
            dataset = PandasDataset.from_long_dataframe(
                y, target=target_col, item_id=item_id, future_length=0, freq=freq
            )

        else:
            self._df_config = {
                "target": [target_col],
            }
            # For single series, infer frequency and pass explicitly
            # to handle both datetime and integer indices
            freq = self.infer_freq(y.index)
            if freq is not None:
                dataset = PandasDataset(
                    y, future_length=0, target=target_col, freq=freq
                )
            else:
                # If frequency cannot be inferred (e.g., integer index),
                # let PandasDataset try without explicit freq
                dataset = PandasDataset(y, future_length=0, target=target_col)

        return dataset

    def _convert_to_float(self, df):
        for col in df.columns:
            # Check if column is not of string type
            if df[col].dtype != "object" and not pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype("float32")

        return df

    def _prepare_training_data(self, y, fh):
        """Prepare training data for fine-tuning.

        Parameters
        ----------
        y : pd.DataFrame
            Training time series data.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        training_data : GluonTS Dataset
            Training dataset in GluonTS format.
        validation_data : GluonTS Dataset or None
            Validation dataset in GluonTS format, or None if validation_split is None.
        """
        from sktime.split import temporal_train_test_split

        # Convert to float
        _y = self._convert_to_float(y.copy())

        # Handle range index - convert to datetime for GluonTS compatibility
        if self.check_range_index(_y):
            _y.index = self.handle_range_index(_y.index)

        # Split into train/validation if needed
        # Now works with hierarchical data thanks to explicit freq parameter
        # in _get_gluonts_dataset
        if self.validation_split is not None and self.validation_split > 0:
            y_train, y_val = temporal_train_test_split(
                _y, test_size=self.validation_split
            )
            training_data = self._get_gluonts_dataset(y_train)
            validation_data = self._get_gluonts_dataset(y_val)
            return training_data, validation_data
        else:
            training_data = self._get_gluonts_dataset(_y)
            return training_data, None

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        For LagLlama, behavior depends on fit_strategy:
        - "zero-shot": Creates predictor from pretrained weights without training
        - "finetuning": Fine-tunes the model on provided data using PyTorch Lightning

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
        # Store the inner column names seen during fit.
        # For pd.Series input, sktime converts to pd.DataFrame with:
        # - column 0 if Series.name is None
        # - column == Series.name otherwise
        # Using y.columns here ensures our internal returns are compatible with
        # sktime's back-conversion (via self._converter_store_y).
        self._fit_column_names = list(y.columns)

        _check_soft_dependencies("torch", severity="error")
        import torch

        # LagLlama does not support in-sample forecasting (fh <= 0)
        fh_rel = fh.to_relative(self.cutoff)
        if len(fh_rel) > 0 and max(fh_rel) <= 0:
            raise NotImplementedError(
                "in-sample forecasting is not supported by LagLlamaForecaster"
            )

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
            lr=self.lr,
            aug_prob=self.aug_prob,
            trainer_kwargs=self._trainer_kwargs,
            nonnegative_pred_samples=self.nonnegative_pred_samples,
        )

        # Create predictor with PyTorch 2.6+ compatibility patch
        # Lightning uses weights_only=True by default which causes issues
        original_load = torch.load

        def patched_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = patched_load
        try:
            if self.fit_strategy == "zero-shot":
                # Zero-shot: create predictor without training
                lightning_module = self.estimator_.create_lightning_module()
                transformation = self.estimator_.create_transformation()
                self.predictor_ = self.estimator_.create_predictor(
                    transformation, lightning_module
                )
            elif self.fit_strategy == "finetuning":
                # Fine-tuning: train the model on provided data
                # Prepare training data
                training_data, validation_data = self._prepare_training_data(y, fh)

                # Train and get predictor
                if validation_data is not None:
                    self.predictor_ = self.estimator_.train(
                        training_data=training_data,
                        validation_data=validation_data,
                        cache_data=True,
                        shuffle_buffer_length=1000,
                    )
                else:
                    self.predictor_ = self.estimator_.train(
                        training_data=training_data,
                        cache_data=True,
                        shuffle_buffer_length=1000,
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
        # Extend the index to cover all out-of-sample points required by fh.

        fh_rel = fh.to_relative(self.cutoff)
        max_step = int(max(fh_rel)) if len(fh_rel) > 0 else 0

        if max_step <= 0:
            pred_index = index[:0]
        elif self.check_range_index(df):
            cutoff_val = (
                self.cutoff[0] if isinstance(self.cutoff, pd.Index) else self.cutoff
            )
            pred_index = pd.RangeIndex(cutoff_val + 1, cutoff_val + max_step + 1)
        elif isinstance(index, pd.PeriodIndex):
            pred_index = pd.period_range(
                self.cutoff[0],
                periods=max_step + 1,
                freq=index.freq,
            )[1:]
        else:
            pred_index = pd.date_range(
                self.cutoff[0],
                periods=max_step + 1,
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

        # LagLlama does not support in-sample forecasting (fh <= 0)
        fh_rel = fh.to_relative(self.cutoff)
        if len(fh_rel) > 0 and min(fh_rel) <= 0:
            raise NotImplementedError(
                "in-sample forecasting is not supported by LagLlamaForecaster"
            )

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
        # make_evaluation_predictions uses sampling; ensure deterministic output across
        # repeated calls (e.g., predict() then score()) by fixing RNG locally.
        import numpy as np
        import torch

        np_state = np.random.get_state()
        np.random.seed(0)
        try:
            with torch.random.fork_rng():
                torch.manual_seed(0)
                forecast_it, _ = make_evaluation_predictions(
                    dataset=dataset,
                    predictor=self.predictor_,
                    num_samples=self.num_samples,
                )
        finally:
            np.random.set_state(np_state)
        predictions = self._get_prediction_df(forecast_it, self._df_config)

        # Convert back to hierarchical if needed
        if _is_hierarchical:
            predictions = self._convert_panel_to_hierarchical(
                predictions, _original_index_names
            )

        # Get the expected prediction index based on cutoff and fh
        # Use the original y (not _y) to get the correct index structure
        pred_out_expected = fh.get_expected_pred_idx(y, cutoff=self.cutoff)

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

            # Subset/align to fh, same as non-range branch
            try:
                predictions = predictions.loc[pred_out_expected]
            except (KeyError, IndexError):
                predictions = predictions.reindex(pred_out_expected)
            predictions.index = pred_out_expected
        else:
            # For non-range indices, align predictions to the expected index
            # This ensures correct timestamps when y has different timestamps
            # than training data
            pred_out_for_loc = pred_out_expected
            if isinstance(predictions.index, pd.PeriodIndex) and isinstance(
                pred_out_expected, pd.DatetimeIndex
            ):
                pred_out_for_loc = pred_out_expected.to_period(predictions.index.freq)
            elif isinstance(predictions.index, pd.DatetimeIndex) and isinstance(
                pred_out_expected, pd.PeriodIndex
            ):
                pred_out_for_loc = pred_out_expected.to_timestamp()

            try:
                predictions = predictions.loc[pred_out_for_loc]
            except (KeyError, IndexError):
                # If direct indexing fails, use reindex to align
                predictions = predictions.reindex(pred_out_for_loc)
            predictions.index = pred_out_expected

        return predictions

    def predict(self, fh=None, X=None, y=None):
        """Forecast time series at future horizon.

        This override caches the last prediction so that `score` can reuse it when
        the passed `y` matches the cached prediction index (avoids re-sampling).
        """
        y_pred = super().predict(fh=fh, X=X, y=y)
        self._last_y_pred_ = y_pred
        return y_pred

    def score(self, y, X=None, fh=None):
        """Scores forecast against ground truth, using MAPE (non-symmetric).

        If `predict` has been called immediately before `score` and the cached
        prediction index matches `y.index`, reuse the cached prediction to avoid
        re-sampling stochastic forecasts.
        """
        from sktime.performance_metrics.forecasting import (
            mean_absolute_percentage_error,
        )

        y_pred = getattr(self, "_last_y_pred_", None)
        if X is None and y_pred is not None and hasattr(y_pred, "index"):
            try:
                if y_pred.index.equals(y.index):
                    return mean_absolute_percentage_error(y, y_pred, symmetric=False)
            except Exception as e:
                # Continue to fallback if index comparison fails
                print(f"Error comparing prediction indices: {e}")
                pass

        return super().score(y, X=X, fh=fh)

    def _predict_quantiles(self, fh, X=None, alpha=None, y=None):
        """Compute quantile forecasts.

        Parameters
        ----------
        fh : ForecastingHorizon
            The forecasting horizon.
        X : pd.DataFrame, optional (default=None)
            Exogenous time series (ignored).
        alpha : list of float, optional (default=None)
            The quantiles to predict. If None, uses default [0.1, 0.25, 0.5, 0.75, 0.9].
        y : pd.DataFrame, optional (default=None)
            Time series for global forecasting. If provided, used instead of self._y.

        Returns
        -------
        quantiles : pd.DataFrame
            Quantile forecasts with MultiIndex (alpha, time) or
            MultiIndex (alpha, item, time) for panel data.
        """
        from gluonts.evaluation import make_evaluation_predictions

        # LagLlama does not support in-sample forecasting (fh <= 0)
        fh_rel = fh.to_relative(self.cutoff)
        if len(fh_rel) > 0 and min(fh_rel) <= 0:
            raise NotImplementedError(
                "in-sample forecasting is not supported by LagLlamaForecaster"
            )

        if alpha is None:
            alpha = [0.1, 0.25, 0.5, 0.75, 0.9]

        # Get the data (use provided y or fall back to self._y)
        if y is None:
            y = self._y
        _y = y.copy()

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
        # ensure deterministic output across repeated calls
        import numpy as np
        import torch

        np_state = np.random.get_state()
        np.random.seed(0)
        try:
            with torch.random.fork_rng():
                torch.manual_seed(0)
                forecast_it, _ = make_evaluation_predictions(
                    dataset=dataset,
                    predictor=self.predictor_,
                    num_samples=self.num_samples,
                )
        finally:
            np.random.set_state(np_state)

        forecasts = list(forecast_it)

        # Extract quantiles for each forecast
        quantile_dfs = []

        for forecast in forecasts:
            # GluonTS forecasts have .quantile(q) method
            forecast_quantiles = {}
            for q in alpha:
                q_val = forecast.quantile(q)
                # Convert to Series if it's a numpy array
                if not isinstance(q_val, pd.Series):
                    q_val = pd.Series(q_val, index=forecast.mean_ts.index)
                forecast_quantiles[q] = q_val

            # Build DataFrame for this forecast
            if forecast.item_id is not None:
                # Panel data - need MultiIndex (alpha, item_id, timepoints)
                for q in alpha:
                    q_series = forecast_quantiles[q]
                    df = q_series.reset_index()
                    df.columns = [self._df_config["timepoints"], "quantile"]
                    df["alpha"] = q
                    df[self._df_config["item_id"]] = forecast.item_id
                    quantile_dfs.append(df)
            else:
                # Single series - MultiIndex (alpha, timepoints)
                for q in alpha:
                    q_series = forecast_quantiles[q]
                    df = q_series.to_frame(name="quantile")
                    df["alpha"] = q
                    df = df.reset_index()
                    df.columns = ["timepoints", "quantile", "alpha"]
                    quantile_dfs.append(df)

        # Combine all quantile forecasts
        if len(quantile_dfs) > 0:
            result = pd.concat(quantile_dfs, ignore_index=True)

            # Set appropriate index for sktime format
            # sktime expects: Index=timepoints, Columns=MultiIndex(variable, alpha)
            # variable names should follow sktime's internal feature naming:
            # - unnamed pd.Series -> variable name 0
            # - named pd.Series -> that name
            # - pd.DataFrame -> column names
            var_name = None
            if hasattr(self, "_y_metadata") and "feature_names" in self._y_metadata:
                # BaseForecaster sets this during fit; for unnamed Series this is [0]
                featnames = self._y_metadata.get("feature_names", None)
                if isinstance(featnames, (list, tuple)) and len(featnames) > 0:
                    var_name = featnames[0]
            if var_name is None:
                # fallback to stored fit columns (may be None for Series name=None)
                var_name = (
                    self._fit_column_names[0]
                    if hasattr(self, "_fit_column_names")
                    and len(self._fit_column_names) > 0
                    else 0
                )
            if forecasts[0].item_id is not None:
                # Panel: MultiIndex (alpha, item_id, timepoints)
                result = result.set_index(
                    ["alpha", self._df_config["item_id"], self._df_config["timepoints"]]
                )
                # Unstack to get: Index=(item_id, timepoints), Columns=alpha
                result = result["quantile"].unstack(level=0)
                # Add variable level to columns using from_tuples
                new_columns = [(var_name, alpha_val) for alpha_val in result.columns]
                result.columns = pd.MultiIndex.from_tuples(
                    new_columns, names=["variable", "alpha"]
                )
            else:
                # Single series: set index to (alpha, timepoints), then unstack
                result = result.set_index(["alpha", "timepoints"])
                # Unstack to get: Index=timepoints, Columns=alpha
                result = result["quantile"].unstack(level=0)
                # Add variable level to columns using from_tuples
                new_columns = [(var_name, alpha_val) for alpha_val in result.columns]
                result.columns = pd.MultiIndex.from_tuples(
                    new_columns, names=["variable", "alpha"]
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

            # Align predictions with expected index based on fh and cutoff
            # Similar to _predict, calculate expected index and align
            if not isinstance(fh.to_pandas(), pd.RangeIndex):
                pred_out_expected = fh.get_expected_pred_idx(y, cutoff=self.cutoff)

                # We want to RETURN the expected index type (derived from y),
                # but may need a converted version to select/reindex safely.
                pred_out_for_loc = pred_out_expected
                if isinstance(result.index, pd.PeriodIndex) and isinstance(
                    pred_out_expected, pd.DatetimeIndex
                ):
                    pred_out_for_loc = pred_out_expected.to_period(result.index.freq)
                elif isinstance(result.index, pd.DatetimeIndex) and isinstance(
                    pred_out_expected, pd.PeriodIndex
                ):
                    pred_out_for_loc = pred_out_expected.to_timestamp()

                # Align the result index using the loc-friendly index
                try:
                    result = result.loc[pred_out_for_loc]
                except (KeyError, IndexError):
                    result = result.reindex(pred_out_for_loc)

                # Explicitly set the final index to the expected sktime index
                result.index = pred_out_expected

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
                "fit_strategy": "zero-shot",
            },
            {
                "context_length": 64,
                "num_samples": 20,
                "use_rope_scaling": True,
                "fit_strategy": "zero-shot",
            },
            {
                "context_length": 32,
                "num_samples": 10,
                "fit_strategy": "finetuning",
                "validation_split": 0.2,
                "trainer_kwargs": {"max_epochs": 1},  # Minimal epochs for testing
                "lr": 5e-4,
            },
        ]

        return params

    def _get_prediction_df(self, forecast_iter, df_config):
        def handle_series_prediction(forecast, df_config):
            # Renames the predicted column to the target column name
            pred = forecast.mean_ts
            # Use the column name from fit data if available
            # This ensures predictions match the original data structure
            if hasattr(self, "_fit_column_names") and len(self._fit_column_names) > 0:
                target_name = self._fit_column_names[0]
            else:
                target_name = df_config["target"][0]
            # Return as DataFrame; BaseForecaster will convert back to pd.Series
            # with correct name using self._converter_store_y (if original
            # input was Series).
            return pred.to_frame(name=target_name)

        def handle_panel_predictions(forecasts_it, df_config):
            # Convert all panel forecasts to a single panel dataframe
            # Use the column name from fit data if available
            if hasattr(self, "_fit_column_names") and len(self._fit_column_names) > 0:
                target_name = self._fit_column_names[0]
            else:
                target_name = df_config["target"][0]
            panels = []
            for forecast in forecasts_it:
                df = forecast.mean_ts.reset_index()
                df.columns = [df_config["timepoints"], target_name]
                df[df_config["item_id"]] = forecast.item_id
                df.set_index(
                    [df_config["item_id"], df_config["timepoints"]], inplace=True
                )
                panels.append(df)
            return pd.concat(panels)

        forecasts = list(forecast_iter)

        # Assuming all forecasts_it are either series or panel type.
        if forecasts[0].item_id is None:
            return handle_series_prediction(forecasts[0], df_config)
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
