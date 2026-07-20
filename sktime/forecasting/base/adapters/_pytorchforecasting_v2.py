# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for pytorch-forecasting v2 models.

This module provides a base adapter class for pytorch-forecasting v2 models,
which use the new D1/D2 data layer (TimeSeries + TslibDataModule) and the
TslibBaseModel model hierarchy.

This is a parallel track to the existing v1 adapter in
``_pytorchforecasting.py`` and does NOT modify or replace it.

Key differences from v1:
- Uses ``TimeSeries`` (D1) instead of ``TimeSeriesDataSet``
- Uses ``TslibDataModule`` (D2) instead of manual dataloader creation
- Models are constructed directly (not via ``from_dataset``)
- Training uses ``trainer.fit(model, datamodule=...)`` pattern
- Parameter layering uses ``data_module_params`` instead of
  ``dataset_params`` + ``train/validation_to_dataloader_params``
"""

import abc
import functools
from copy import deepcopy

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sktime.forecasting.base import (
    BaseForecaster,
    ForecastingHorizon,
)

__all__ = ["_PytorchForecastingAdapterV2"]
__author__ = ["vedantag17"]


def _series_to_frame(data):
    """Convert pd.Series to pd.DataFrame if needed.

    Parameters
    ----------
    data : pd.Series, pd.DataFrame, or None

    Returns
    -------
    _data : pd.DataFrame or None
    converted : bool
        True if conversion happened
    """
    converted = False
    if data is not None:
        if isinstance(data, pd.Series):
            _data = deepcopy(data).to_frame(name=data.name)
            converted = True
        else:
            _data = deepcopy(data)
    else:
        _data = None
    return _data, converted


def _sktime_to_ptf_v2_timeseries(
    y,
    X=None,
    time_col="_auto_time_idx",
    target_col="_target",
    group_cols=None,
):
    """Convert sktime panel/hierarchical data to PTF v2 ``TimeSeries`` D1 dataset.

    This is the core data conversion function for the v2 adapter.
    It is designed to be unit-testable independent of any model.

    Parameters
    ----------
    y : pd.DataFrame
        Target data. May have a MultiIndex (panel/hierarchical) or single index.
        Must have exactly one column (the target).
    X : pd.DataFrame or None
        Exogenous features, same index structure as ``y``.
    time_col : str, default="_auto_time_idx"
        Name for the synthetic integer time index column.
    target_col : str, default="_target"
        Name to use for the target column in the output.
    group_cols : list of str or None
        If None, will be inferred from the index structure.

    Returns
    -------
    timeseries : pytorch_forecasting.data.timeseries.TimeSeries
        A v2 TimeSeries D1 dataset object.
    metadata : dict
        Metadata about the conversion, including:
        - ``target_name``: original target column name
        - ``index_names``: original index level names
        - ``index_len``: number of index levels
        - ``x_columns``: original X column names (if X is not None)
        - ``group_cols``: list of group column names used
        - ``time_col``: name of the time column used
        - ``convert_to_series``: whether the input y was a pd.Series
    """
    from pytorch_forecasting.data.timeseries._timeseries_v2 import TimeSeries

    # Store original metadata for reverse conversion
    target_name = y.columns[0]
    index_names = list(y.index.names)
    index_len = len(index_names)
    x_columns = list(X.columns) if X is not None else []

    conversion_meta = {
        "target_name": target_name,
        "index_names": index_names,
        "index_len": index_len,
        "x_columns": x_columns,
        "time_col": time_col,
        "target_col": target_col,
    }

    # Rename target column to a safe string name
    y = y.copy()
    y.columns = [target_col]

    # Rename X columns to safe string names
    if X is not None:
        X = X.copy()
        new_x_columns = [f"_x_{i}" for i in range(len(x_columns))]
        X.columns = new_x_columns
        conversion_meta["new_x_columns"] = new_x_columns
    else:
        new_x_columns = []

    # Add integer time index
    if index_len > 1:
        # Panel / hierarchical: groupby all levels except the last (time)
        group_level_names = index_names[:-1]
        time_idx = y.groupby(level=list(range(index_len - 1))).cumcount()
    else:
        group_level_names = []
        time_idx = pd.Series(range(len(y)), index=y.index)

    # Reset index to get group columns as regular columns. Use the actual
    # reset_index column labels because unnamed levels become generated labels
    # such as "level_0", which cannot be recovered from y.index.names.
    y_reset = y.reset_index()
    reset_index_cols = list(y_reset.columns[:index_len])
    reset_group_cols = reset_index_cols[:-1]
    reset_time_col = reset_index_cols[-1]

    if X is not None:
        X_reset = X.reset_index(drop=True)
        data = pd.concat([y_reset, X_reset], axis=1)
    else:
        data = y_reset.copy()

    # Add the time index column
    data[time_col] = time_idx.values

    # Determine group columns
    if index_len > 1:
        # Rename the hierarchical index columns (all except time) to safe
        # string names used by pytorch-forecasting's TimeSeries grouping.
        new_group_cols = [f"_group_{i}" for i in range(len(reset_group_cols))]
        rename_map = {old: new for old, new in zip(reset_group_cols, new_group_cols)}
        data = data.rename(columns=rename_map)
        conversion_meta["group_cols"] = new_group_cols
        conversion_meta["original_group_cols"] = group_level_names
    else:
        # Single time series — add a constant group column
        data["_auto_group"] = 0
        new_group_cols = ["_auto_group"]
        conversion_meta["group_cols"] = new_group_cols
        conversion_meta["original_group_cols"] = []

    # Drop the original time index column if it exists as a regular column
    if reset_time_col in data.columns and reset_time_col != time_col:
        # Save the original time values for later reverse conversion
        conversion_meta["original_time_values"] = data[reset_time_col].copy()
        data = data.drop(columns=[reset_time_col])

    # Ensure all feature columns are float
    for col in new_x_columns:
        if col in data.columns:
            data[col] = data[col].astype(float)
    data[target_col] = data[target_col].fillna(0).astype(float)

    # Identify numeric (continuous) features
    feature_cols = [c for c in new_x_columns if is_numeric_dtype(data[c].dtype)]

    # Build the known/unknown classification
    # In sktime's convention, X variables are "known" (they're exogenous)
    known = list(feature_cols)
    unknown = []

    # Create TimeSeries D1 object
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        timeseries = TimeSeries(
            data=data,
            time=time_col,
            target=[target_col],
            group=new_group_cols if new_group_cols != ["_auto_group"] else None,
            num=feature_cols if feature_cols else None,
            known=known if known else None,
            unknown=unknown if unknown else None,
        )

    return timeseries, conversion_meta


class _PytorchForecastingAdapterV2(BaseForecaster):
    """Base adapter class for pytorch-forecasting v2 models.

    This adapter targets PTF v2's D1/D2 data pipeline (``TimeSeries`` +
    ``TslibDataModule``) and ``TslibBaseModel`` model hierarchy.

    It is a **parallel** adapter to ``_PytorchForecastingAdapter`` (v1) and
    does not modify or replace it.

    Parameters
    ----------
    model_params : dict[str, Any] or None, default=None
        Parameters passed to the PTF v2 model constructor.
        Example: ``{"moving_avg": 25, "individual": False}`` for DLinear.
    data_module_params : dict[str, Any] or None, default=None
        Parameters for ``TslibDataModule``.
        ``context_length`` and ``prediction_length`` will be inferred from data
        and ``fh`` if not provided.
        ``time_series_dataset`` is constructed automatically from sktime data.
        **NOTE**: This replaces the v1 three-way split of ``dataset_params``,
        ``train_to_dataloader_params``, and ``validation_to_dataloader_params``
        because v2's ``TslibDataModule`` manages both datasets and dataloaders
        internally as a ``LightningDataModule``.
    trainer_params : dict[str, Any] or None, default=None
        Parameters for ``lightning.pytorch.Trainer``.
        Example: ``{"max_epochs": 10, "accelerator": "cpu"}``.
    random_log_path : bool, default=False
        Use random root directory for logging.
    broadcasting : bool, default=False
        If True, fall back to per-series fitting.
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["vedantag17"],
        "maintainers": ["vedantag17"],
        "python_dependencies": [
            "pytorch-forecasting>=1.0.0",
            "torch",
            "lightning",
        ],
        # estimator type
        # --------------
        "y_inner_mtype": [
            "pd-multiindex",
            "pd_multiindex_hier",
            "pd.Series",
        ],
        "X_inner_mtype": [
            "pd-multiindex",
            "pd_multiindex_hier",
            "pd.DataFrame",
        ],
        "capability:multivariate": False,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "property:randomness": "stochastic",
        # CI and testing tags
        # -------------------
        "tests:vm": False,
        "tests:libs": [
            "sktime.forecasting.base.adapters._pytorchforecasting_v2",
        ],
        "tests:skip_by_name": [
            "test_save_estimators_to_file",
            "test_persistence_via_pickle",
        ],
    }

    def __init__(
        self,
        model_params=None,
        data_module_params=None,
        trainer_params=None,
        random_log_path=False,
        broadcasting=False,
    ):
        self.model_params = model_params
        self.data_module_params = data_module_params
        self.trainer_params = trainer_params
        self.random_log_path = random_log_path
        self.broadcasting = broadcasting

        # Internal deep copies to avoid mutation
        self._model_params = deepcopy(model_params) if model_params else {}
        self._data_module_params = (
            deepcopy(data_module_params) if data_module_params else {}
        )
        self._model_loss = self._model_params.pop("loss", None)
        self._callbacks = (
            deepcopy(trainer_params).pop("callbacks", None) if trainer_params else None
        )
        self._trainer_params = deepcopy(trainer_params) if trainer_params else {}
        # Remove callbacks from trainer_params copy (will be passed separately)
        self._trainer_params.pop("callbacks", None)

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.Series",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )
        super().__init__()

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_class(self):
        """Import underlying pytorch-forecasting v2 algorithm class."""

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_parameters(self) -> dict:
        """Keyword parameters for the underlying algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Parameters
        ----------
        y : sktime time series object
            Target time series to fit to.
        fh : ForecastingHorizon
            The forecasting horizon with the steps ahead to predict.
        X : sktime time series object, optional (default=None)
            Exogenous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self._max_prediction_length = int(np.max(fh.to_relative(self.cutoff)))

        if not fh.is_all_out_of_sample(self.cutoff):
            raise NotImplementedError(
                f"No in sample predict support, but found fh with in sample index: {fh}"
            )

        # Convert series to frame
        _y, self._convert_to_series = _series_to_frame(y)
        _X, _ = _series_to_frame(X)

        # Determine context_length from data or params
        context_length = self._data_module_params.get("context_length", None)
        if context_length is None:
            # Use a sensible default: min(max available, 2 * prediction_length)
            if _y.index.nlevels > 1:
                group_sizes = _y.groupby(level=list(range(_y.index.nlevels - 1))).size()
                min_series_len = group_sizes.min()
            else:
                min_series_len = len(_y)
            context_length = min(
                min_series_len - self._max_prediction_length,
                2 * self._max_prediction_length,
            )
            context_length = max(context_length, 1)

        self._context_length = context_length

        # Convert sktime data to PTF v2 TimeSeries (D1)
        timeseries, self._conversion_meta = _sktime_to_ptf_v2_timeseries(_y, _X)

        # Build TslibDataModule (D2)
        self._data_module = self._build_data_module(
            timeseries, context_length, self._max_prediction_length
        )

        # Setup the data module
        self._data_module.setup(stage="fit")

        # Instantiate model
        self._forecaster = self._instantiate_model(self._data_module)

        # Instantiate trainer
        import lightning.pytorch as pl

        trainer_kwargs = deepcopy(self._trainer_params)
        if self.random_log_path:
            if (
                "logger" not in trainer_kwargs
                and "default_root_dir" not in trainer_kwargs
            ):
                import os
                import time
                from random import randint

                random_num = (
                    hash(time.time_ns())
                    + hash(self.algorithm_class)
                    + hash(randint(0, int(time.time())))  # noqa: S311
                )
                self._random_log_dir = (
                    os.getcwd() + "/lightning_logs/" + str(abs(random_num))
                )
                trainer_kwargs["default_root_dir"] = self._random_log_dir

        self._trainer = pl.Trainer(callbacks=self._callbacks, **trainer_kwargs)

        # Train
        self._trainer.fit(self._forecaster, datamodule=self._data_module)

        self.best_model = self._forecaster
        return self

    def _predict(self, fh=None, X=None):
        """Forecast time series at future horizon.

        Parameters
        ----------
        fh : ForecastingHorizon or None
            The forecasting horizon.
        X : sktime time series object, optional (default=None)
            Exogenous time series for the forecast.

        Returns
        -------
        y_pred : pd.DataFrame or pd.Series
            Point predictions.
        """
        # For prediction, we need to rebuild the data with the full
        # history plus the prediction window
        y = deepcopy(self._y)
        if X is None:
            X = deepcopy(self._X)
        elif self._X is not None:
            X = pd.concat([deepcopy(self._X), X])

        _y, _ = _series_to_frame(y)
        _X, _ = _series_to_frame(X)

        # Extend y into the future
        _y = self._extend_y(_y, fh)

        # Rebuild TimeSeries and DataModule for prediction
        timeseries, _ = _sktime_to_ptf_v2_timeseries(_y, _X)

        pred_data_module = self._build_data_module(
            timeseries, self._context_length, self._max_prediction_length
        )
        pred_data_module.setup(stage="predict")

        # Get predictions
        import lightning.pytorch as pl

        trainer_kwargs = {}
        if hasattr(self, "_random_log_dir"):
            trainer_kwargs["default_root_dir"] = self._random_log_dir
        trainer_kwargs["logger"] = False

        pred_trainer = pl.Trainer(**trainer_kwargs)
        predictions = pred_trainer.predict(self.best_model, datamodule=pred_data_module)

        # Convert predictions back to sktime format
        output = self._predictions_to_dataframe(predictions, fh)
        return output

    def _build_data_module(self, timeseries, context_length, prediction_length):
        """Build a TslibDataModule from a TimeSeries D1 dataset.

        Parameters
        ----------
        timeseries : pytorch_forecasting.data.timeseries.TimeSeries
            The D1 dataset.
        context_length : int
            Number of past time steps for the encoder.
        prediction_length : int
            Number of future time steps for the decoder.

        Returns
        -------
        data_module : TslibDataModule
            The D2 data module ready for setup().
        """
        import warnings

        from pytorch_forecasting.data.data_module._tslib_data_module import (
            TslibDataModule,
        )

        dm_params = deepcopy(self._data_module_params)
        # Override context_length and prediction_length
        dm_params["context_length"] = context_length
        dm_params["prediction_length"] = prediction_length
        dm_params["time_series_dataset"] = timeseries

        # Set sensible defaults for training efficiency
        dm_params.setdefault("batch_size", 32)
        dm_params.setdefault("num_workers", 0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            data_module = TslibDataModule(**dm_params)

        return data_module

    def _instantiate_model(self, data_module):
        """Instantiate the PTF v2 model.

        Parameters
        ----------
        data_module : TslibDataModule
            The data module providing metadata.

        Returns
        -------
        model : TslibBaseModel subclass
            The instantiated model.
        """
        from pytorch_forecasting.metrics import MAE

        model_params = deepcopy(self._model_params)
        model_params.update(self.algorithm_parameters)

        # Set loss — default to MAE if none provided
        loss = self._model_loss
        if loss is None:
            loss = MAE()
        model_params["loss"] = loss

        # Pass metadata from data module
        model_params["metadata"] = data_module.metadata

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            model = self.algorithm_class(**model_params)

        return model

    def _predictions_to_dataframe(self, predictions, fh):
        """Convert PTF v2 predictions to sktime DataFrame.

        Parameters
        ----------
        predictions : list of dicts
            Output from ``trainer.predict()``, each dict has "prediction" tensor.
        fh : ForecastingHorizon
            The forecasting horizon.

        Returns
        -------
        output : pd.DataFrame or pd.Series
            Predictions in sktime format.
        """
        import torch

        # Collect all prediction tensors
        all_preds = []
        for batch_result in predictions:
            pred = batch_result["prediction"]
            if isinstance(pred, torch.Tensor):
                all_preds.append(pred.cpu().detach().numpy())

        if not all_preds:
            raise ValueError("No predictions were generated.")

        preds = np.concatenate(all_preds, axis=0)

        # preds shape: (n_windows, prediction_length, n_targets) or
        #              (n_windows, prediction_length)
        if preds.ndim == 3 and preds.shape[-1] == 1:
            preds = preds.squeeze(-1)

        # Build the output index from the forecasting horizon
        meta = self._conversion_meta
        target_name = meta["target_name"]
        absolute_horizons = fh.to_absolute_index(self.cutoff)

        # For panel/hierarchical data, we need to reconstruct the full index
        if meta["index_len"] > 1:
            # Get the unique group combinations from the original y
            y_frame = self._y
            if isinstance(y_frame, pd.Series):
                y_frame = y_frame.to_frame()

            group_levels = list(range(meta["index_len"] - 1))
            groups = y_frame.groupby(level=group_levels).groups
            unique_groups = list(groups.keys())

            # We expect one prediction window per group
            # Take mean if multiple windows exist per group, or last window

            # Build multi-index
            tuples = []
            values = []
            for i, group_key in enumerate(unique_groups):
                if not isinstance(group_key, tuple):
                    group_key = (group_key,)
                for j, time_val in enumerate(absolute_horizons):
                    tuples.append((*group_key, time_val))
                    # Use the last prediction window for this group
                    if i < preds.shape[0]:
                        if j < preds.shape[1]:
                            values.append(preds[i, j])
                        else:
                            values.append(np.nan)
                    else:
                        values.append(np.nan)

            idx = pd.MultiIndex.from_tuples(tuples, names=meta["index_names"])
            output = pd.DataFrame({target_name: values}, index=idx)
        else:
            # Single time series
            pred_values = preds[0] if preds.ndim > 1 else preds
            pred_values = pred_values[: len(absolute_horizons)]

            output = pd.DataFrame(
                {target_name: pred_values},
                index=absolute_horizons,
            )
            output.index.name = meta["index_names"][0]

        if self._convert_to_series:
            output = pd.Series(
                data=output[target_name],
                index=output.index,
                name=target_name,
            )

        return output

    def _extend_y(self, y, fh):
        """Extend y into the future with placeholder values.

        Parameters
        ----------
        y : pd.DataFrame
            Historical target data.
        fh : ForecastingHorizon
            Forecasting horizon.

        Returns
        -------
        extended_y : pd.DataFrame
            y with future time indices appended (filled with 0).
        """
        _fh = ForecastingHorizon(
            range(1, self._max_prediction_length + 1),
            is_relative=True,
            freq=fh.freq,
        )
        index = _fh.to_absolute_index(self.cutoff)
        _y = pd.DataFrame(index=index, columns=y.columns)
        _y.index.rename(y.index.names[-1], inplace=True)
        _y.fillna(0, inplace=True)
        len_levels = len(y.index.names)
        if len_levels == 1:
            _y = pd.concat([y, _y])
        else:
            _y = y.groupby(level=list(range(len_levels - 1))).apply(
                lambda x: pd.concat([x.droplevel(list(range(len_levels - 1))), _y])
            )
        return _y
