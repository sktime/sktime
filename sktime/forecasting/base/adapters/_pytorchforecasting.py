# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for pytorch-forecasting models."""

import abc
import functools
import inspect
import os
import time
import typing
from copy import deepcopy
from random import randint
from typing import Any, Optional

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from sktime.forecasting.base import ForecastingHorizon, _BaseGlobalForecaster

__all__ = ["_PytorchForecastingAdapter"]
__author__ = ["XinyuWu"]


class _PytorchForecastingAdapter(_BaseGlobalForecaster):
    """Base adapter class for pytorch-forecasting models.

    Parameters
    ----------
    model_params :  Dict[str, Any] (default=None)
        parameters to be passed to initialize the underlying pytorch-forecasting model
        for example: {"lstm_layers": 3, "hidden_continuous_size": 10} for TFT model
    dataset_params : Dict[str, Any] (default=None)
        parameters to initialize `TimeSeriesDataSet` [1]_ from `pandas.DataFrame`
        max_prediction_length will be overwrite according to fh
        time_idx, target, group_ids, time_varying_known_reals, time_varying_unknown_reals
        will be inferred from data, so you do not have to pass them
    train_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": True}
    validation_to_dataloader_params : Dict[str, Any] (default=None)
        parameters to be passed for `TimeSeriesDataSet.to_dataloader()`
        by default {"train": False}
    model_path: string (default=None)
        try to load a existing model without fitting. Calling the fit function is
        still needed, but no real fitting will be performed.
    random_log_path: bool (default=False)
        use random root directory for logging. This parameter is for CI test in
        Github action, not designed for end users.

    References
    ----------
    .. [1] https://pytorch-forecasting.readthedocs.io/en/stable/api/pytorch_forecasting.data.timeseries.TimeSeriesDataSet.html
    """  # noqa: E501

    _tags = {
        # packaging info
        # --------------
        "authors": ["XinyuWu"],
        "maintainers": ["XinyuWu"],
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
        "scitype:y": "univariate",
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "capability:missing_values": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
    }

    def __init__(
        self: "_PytorchForecastingAdapter",
        model_params: Optional[dict[str, Any]] = None,
        dataset_params: Optional[dict[str, Any]] = None,
        train_to_dataloader_params: Optional[dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[dict[str, Any]] = None,
        trainer_params: Optional[dict[str, Any]] = None,
        model_path: Optional[str] = None,
        random_log_path: bool = False,
        broadcasting: bool = False,
    ) -> None:
        self.model_params = model_params
        self.dataset_params = dataset_params
        self.trainer_params = trainer_params
        self.train_to_dataloader_params = train_to_dataloader_params
        self.validation_to_dataloader_params = validation_to_dataloader_params
        self.model_path = model_path
        self._model_params = deepcopy(model_params) if model_params is not None else {}
        self._dataset_params = (
            deepcopy(dataset_params) if dataset_params is not None else {}
        )
        self._trainer_params = (
            deepcopy(trainer_params) if trainer_params is not None else {}
        )
        self._train_to_dataloader_params = (
            deepcopy(train_to_dataloader_params)
            if train_to_dataloader_params is not None
            else {}
        )
        self._validation_to_dataloader_params = (
            deepcopy(validation_to_dataloader_params)
            if validation_to_dataloader_params is not None
            else {}
        )
        self.random_log_path = random_log_path
        self.broadcasting = broadcasting
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
    def algorithm_class(self: "_PytorchForecastingAdapter"):
        """Import underlying pytorch-forecasting algorithm class."""

    @functools.cached_property
    @abc.abstractmethod
    def algorithm_parameters(self: "_PytorchForecastingAdapter") -> dict:
        """Keyword parameters for the underlying pytorch-forecasting algorithm class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class

        """

    def _instantiate_model(self: "_PytorchForecastingAdapter", data):
        """Instantiate the model."""
        algorithm_instance = self.algorithm_class.from_dataset(
            data,
            **self.algorithm_parameters,
            **self._model_params,
        )
        import lightning.pytorch as pl

        if self.random_log_path:
            if "logger" not in self._trainer_params.keys():
                if "default_root_dir" not in self._trainer_params.keys():
                    self._random_log_dir = self._gen_random_log_dir(data)
                    self._trainer_params["default_root_dir"] = self._random_log_dir

        trainer_instance = pl.Trainer(**self._trainer_params)
        return algorithm_instance, trainer_instance

    def _gen_random_log_dir(self, data=None):
        random_num = (
            hash(time.time_ns())
            + hash(self.algorithm_class)
            + hash(str(data.get_parameters()) if data else "NoDataPassed")
            + hash(randint(0, int(time.time())))  # noqa: S311
        )
        random_log_dir = os.getcwd() + "/lightning_logs/" + str(abs(random_num))
        return random_log_dir

    def _fit(
        self: "_PytorchForecastingAdapter",
        y: pd.DataFrame,
        X: typing.Optional[pd.DataFrame],
        fh: ForecastingHorizon,
    ) -> "_PytorchForecastingAdapter":
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to have a single column/variable
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : sktime time series object, optional (default=None)
            guaranteed to have at least one column/variable
            Exogeneous time series to fit to.

        Returns
        -------
        self : _PytorchForecastingAdapter
            reference to self
        """
        self._max_prediction_length = np.max(fh.to_relative(self.cutoff))
        if not fh.is_all_out_of_sample(self.cutoff):
            raise NotImplementedError(
                f"No in sample predict support, but found fh with in sample index: {fh}"
            )
        # check if dummy X is needed
        # only the TFT model need X to fit, probably a bug in pytorch-forecasting
        X = self._dummy_X(X, y)
        # convert series to frame
        _y, self._convert_to_series = _series_to_frame(y)
        _X, _ = _series_to_frame(X)
        # convert data to pytorch-forecasting datasets
        training, validation = self._Xy_to_dataset(
            _X, _y, self._dataset_params, self._max_prediction_length
        )
        if self.model_path is None:
            # instantiate forecaster and trainer
            self._forecaster, self._trainer = self._instantiate_model(training)
            # convert dataset to dataloader
            self._train_to_dataloader_params["train"] = True
            self._validation_to_dataloader_params["train"] = False
            # call the fit function of the pytorch-forecasting model
            self._trainer.fit(
                self._forecaster,
                train_dataloaders=training.to_dataloader(
                    **self._train_to_dataloader_params
                ),
                val_dataloaders=validation.to_dataloader(
                    **self._validation_to_dataloader_params
                ),
            )
            if self._trainer.checkpoint_callback is not None:
                # load model from checkpoint
                best_model_path = self._trainer.checkpoint_callback.best_model_path
                self.best_model = self.algorithm_class.load_from_checkpoint(
                    best_model_path
                )
            else:
                self.best_model = self._forecaster
        else:
            # load model from disk
            self.best_model = self.algorithm_class.load_from_checkpoint(self.model_path)
        return self

    def _predict(
        self: "_PytorchForecastingAdapter",
        fh: typing.Optional[ForecastingHorizon],
        X: typing.Optional[pd.DataFrame],
        y: typing.Optional[pd.DataFrame],
    ) -> pd.Series:
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
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
            If ``y`` is not passed (not performing global forecasting), ``X`` should
            only contain the time points to be predicted.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.
        y : sktime time series object, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        y_pred : sktime time series object
            guaranteed to have a single column/variable
            Point predictions

        Notes
        -----
        If ``y`` is not None, global forecast will be performed.
        In global forecast mode,
        ``X`` should contain all historical values and the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.

        If ``y`` is None, non global forecast will be performed.
        In non global forecast mode,
        ``X`` should only contain the time points to be predicted,
        while ``y`` should only contain historical values
        not the time points to be predicted.
        """
        X, y = self._Xy_precheck(X, y)
        # convert series to frame
        _y, self._convert_to_series = _series_to_frame(y)
        _X, _ = _series_to_frame(X)
        # extend index of y
        _y = self._extend_y(_y, fh)
        # check if dummy X is needed
        _X = self._dummy_X(_X, _y)
        # convert data to pytorch-forecasting datasets
        training, validation = self._Xy_to_dataset(
            _X, _y, self._dataset_params, self._max_prediction_length
        )
        try:
            if self.deterministic:
                import torch

                torch_state = torch.get_rng_state()
                torch.manual_seed(0)
        except AttributeError:
            pass

        predictions = self.best_model.predict(
            validation.to_dataloader(**self._validation_to_dataloader_params),
            return_x=True,
            return_index=True,
            return_decoder_lengths=True,
            trainer_kwargs=(
                {"default_root_dir": self._random_log_dir}
                if "_random_log_dir" in self.__dict__.keys()
                else None
            ),
        )
        try:
            if self.deterministic:
                torch.set_rng_state(torch_state)
        except AttributeError:
            pass
        # convert pytorch-forecasting predictions to dataframe
        output = self._predictions_to_dataframe(
            predictions, self._max_prediction_length
        )

        absolute_horizons = self.fh.to_absolute_index(self.cutoff)
        dateindex = output.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        return output.loc[dateindex]

    def _predict_quantiles(self, fh, X, alpha, y=None):
        """Compute/return prediction quantiles for a forecast.

        private _predict_quantiles containing the core logic,
            called from predict_quantiles and default _predict_interval

        Parameters
        ----------
        fh : guaranteed to be ForecastingHorizon
            The forecasting horizon with the steps ahead to to predict.
        X : optional (default=None)
            guaranteed to be of a type in self.get_tag("X_inner_mtype")
            Exogeneous time series to predict from.
            If ``y`` is passed (performing global forecasting), ``X`` must contain
            all historical values and the time points to be predicted.
        alpha : list of float, optional (default=[0.5])
            A list of probabilities at which quantile forecasts are computed.
        y : time series in ``sktime`` compatible format, optional (default=None)
            Historical values of the time series that should be predicted.
            If not None, global forecasting will be performed.
            Only pass the historical values not the time points to be predicted.

        Returns
        -------
        quantiles : pd.DataFrame
            Column has multi-index: first level is variable name from y in fit,
                second level being the values of alpha passed to the function.
            Row index is fh, with additional (upper) levels equal to instance levels,
                    from y seen in fit, if y_inner_mtype is Panel or Hierarchical.
            Entries are quantile forecasts, for var in col index,
                at quantile probability in second col index, for the row index.
        """
        methods_list = [
            method[0]
            for method in inspect.getmembers(
                self.best_model.loss, predicate=inspect.ismethod
            )
        ]
        if "to_quantiles" not in methods_list:
            raise NotImplementedError(
                "To perform probabilistic forecast, QuantileLoss or other loss"
                "metrics that support to_quantiles function has to be used in fit."
                f"With {self.best_model.loss}, it doesn't support probabilistic"
                "forecast. Details can be found:"
                "https://pytorch-forecasting.readthedocs.io/en/stable/metrics.html"
            )

        X, y = self._Xy_precheck(X, y)
        # convert series to frame
        _y, self._convert_to_series = _series_to_frame(y)
        _X, _ = _series_to_frame(X)
        # extend index of y
        _y = self._extend_y(_y, fh)
        # check if dummy X is needed
        _X = self._dummy_X(_X, _y)
        # convert data to pytorch-forecasting datasets
        training, validation = self._Xy_to_dataset(
            _X, _y, self._dataset_params, self._max_prediction_length
        )
        try:
            if self.deterministic:
                import torch

                torch_state = torch.get_rng_state()
                torch.manual_seed(0)
        except AttributeError:
            pass

        predictions = self.best_model.predict(
            validation.to_dataloader(**self._validation_to_dataloader_params),
            mode="quantiles",
            mode_kwargs={"use_metric": False, "quantiles": alpha},
            return_x=True,
            return_index=True,
            return_decoder_lengths=True,
            trainer_kwargs=(
                {"default_root_dir": self._random_log_dir}
                if "_random_log_dir" in self.__dict__.keys()
                else None
            ),
        )
        try:
            if self.deterministic:
                torch.set_rng_state(torch_state)
        except AttributeError:
            pass
        # convert pytorch-forecasting predictions to dataframe
        output = self._predictions_to_dataframe(
            predictions, self._max_prediction_length, alpha=alpha
        )

        absolute_horizons = self.fh.to_absolute_index(self.cutoff)
        dateindex = output.index.get_level_values(-1).map(
            lambda x: x in absolute_horizons
        )
        return output.loc[dateindex]

    def _Xy_precheck(self, X, y):
        if y is None:
            y = deepcopy(self._y)
        if X is None:
            X = deepcopy(self._X)
        if X is not None and not self._global_forecasting:
            X = pd.concat([self._X, X])
        return X, y

    def _Xy_to_dataset(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame,
        dataset_params: dict[str, Any],
        max_prediction_length,
    ):
        from pytorch_forecasting.data import TimeSeriesDataSet

        # X, y must have same index or X is None
        # assert X is None or (X.index == y.index).all()
        # might not the same order
        # store the target column names and index names (probably [None])
        # will be renamed !
        self._target_name = y.columns[-1]
        self._index_names = y.index.names
        self._index_len = len(self._index_names)
        # store X, y column names (probably None or not str type)
        # The target column and the index will be renamed
        # before being passed to the underlying model
        # because those names could be None or non-string type.
        if X is not None:
            self._X_columns = X.columns.tolist()
        # rename the index to make sure it's not None
        self._new_index_names = [
            "_index_name_" + str(i) for i in range(len(self._index_names))
        ]
        if self._index_len == 1:
            rename_input = self._new_index_names[0]
        else:
            rename_input = self._new_index_names
        y.index.rename(rename_input, inplace=True)
        if X is not None:
            X.index.rename(rename_input, inplace=True)
        # rename X, y columns names to make sure they are all str type
        if X is not None:
            self._new_X_columns = [
                "_X_column_" + str(i) for i in range(len(self._X_columns))
            ]
            X.columns = self._new_X_columns
        self._new_target_name = "_target_column"
        y.columns = [self._new_target_name]
        # combine X and y
        if X is not None:
            # only numeric columns
            time_varying_known_reals = [
                c for c in X.columns if is_numeric_dtype(X[c].dtype)
            ]
            data = X.join(y, on=X.index.names)
        else:
            time_varying_known_reals = []
            data = deepcopy(y)
        # if fh is not continuous, there will be NaN after extend_y in prediect
        data = data.copy()
        data["_target_column"] = data["_target_column"].fillna(0)
        # add integer time_idx column as pytorch-forecasting requires
        if self._index_len > 1:
            time_idx = (
                data.groupby(by=self._new_index_names[0:-1]).cumcount().to_frame()
            )
        else:
            time_idx = pd.DataFrame(range(0, len(data)))
            time_idx.index = data.index
        time_idx.rename(columns={0: "_auto_time_idx"}, inplace=True)
        data = pd.concat([data, time_idx], axis=1)
        # reset multi index to normal columns
        data = data.reset_index(level=list(range(self._index_len)))
        training_cutoff = data["_auto_time_idx"].max() - max_prediction_length
        # add a constant column as group id if data only contains only one timeseries
        if self._index_len == 1:
            group_id = pd.DataFrame([0] * len(data))
            group_id.index = data.index
            group_id.rename(columns={0: "_auto_group_id"}, inplace=True)
            data = pd.concat([group_id, data], axis=1)
        # save origin time idx for prediction
        self._origin_time_idx = data[
            (
                ([] if self._index_len > 1 else ["_auto_group_id"])
                + self._new_index_names
                + ["_auto_time_idx"]
            )
        ][data["_auto_time_idx"] > training_cutoff]
        # infer time_idx column, target column and instances from data
        _dataset_params = {
            "data": data[data["_auto_time_idx"] <= training_cutoff],
            "time_idx": "_auto_time_idx",
            "target": self._new_target_name,
            "group_ids": (
                self._new_index_names[0:-1]
                if self._index_len > 1
                else ["_auto_group_id"]
            ),
            "time_varying_known_reals": time_varying_known_reals,
            "time_varying_unknown_reals": [self._new_target_name],
        }
        _dataset_params.update(dataset_params)
        # overwrite max_prediction_length
        _dataset_params["max_prediction_length"] = int(max_prediction_length)
        training = TimeSeriesDataSet(**_dataset_params)
        validation = TimeSeriesDataSet.from_dataset(
            training, data, predict=True, stop_randomization=True
        )
        return training, validation

    def _predictions_to_dataframe(self, predictions, max_prediction_length, alpha=None):
        # output is the actual predictions points but without index
        output = predictions.output.cpu().numpy()
        # index will be combined with output
        index = predictions.index
        # in pytorch-forecasting predictions, the first index is the time_idx
        index_names = index.columns.to_list()
        time_idx = index_names.pop(0)
        # make time_idx the last index
        index_names.append(time_idx)
        # in pytorch-forecasting predictions,
        # the index only contains the start timepoint.
        data = index.loc[index.index.repeat(max_prediction_length)].reset_index(
            drop=True
        )
        # make time_idx the last index
        data = data.reindex(columns=index_names)
        # correct the time_idx after repeating
        # assume the time_idx column is continuous integers
        # it's always true as a new time_idx column is added
        # to the data before it's passed to underlying model
        for i in range(output.shape[0]):
            start_idx = i * max_prediction_length
            start_time = data.loc[start_idx, time_idx]
            data.loc[start_idx : start_idx + max_prediction_length - 1, time_idx] = (
                list(range(start_time, start_time + max_prediction_length))
            )

        # set the instance columns to multi index
        data.set_index(index_names, inplace=True)
        self._origin_time_idx.set_index(index_names, inplace=True)
        # add origin time_idx column to data
        data = data.join(self._origin_time_idx, on=index_names)
        # drop _auto_time_idx column
        data.reset_index(level=list(range(len(index_names))), inplace=True)
        data.drop("_auto_time_idx", axis=1, inplace=True)
        index_names.remove("_auto_time_idx")
        # drop _auto_group_id column
        if self._index_len == 1:
            data.drop("_auto_group_id", axis=1, inplace=True)
            index_names.remove("_auto_group_id")
        # reindex to origin multiindex
        # the last of self._new_index_names is the time index
        data.set_index(
            index_names + [self._new_index_names[-1]],
            inplace=True,
        )
        # set index names back to original input in fit
        if self._index_len == 1:
            rename_input = self._index_names[0]
        else:
            rename_input = self._index_names
        data.index.rename(rename_input, inplace=True)
        # add the target column at the end
        if alpha is not None:
            quantiles = output.reshape((-1, len(alpha)))
            quantiles = pd.DataFrame(
                data=quantiles,
                index=data.index,
            )
            data = pd.concat([data, quantiles], axis=1)
            data.columns = [
                [self._target_name if self._target_name is not None else 0]
                * len(alpha),
                alpha,
            ]
        else:
            data[self._target_name] = output.flatten()
        # # set target name back to original input in fit
        # data.columns = [self._target_name]
        # convert back to pd.series if needed
        if self._convert_to_series and alpha is None:
            data = pd.Series(
                data=data[self._target_name], index=data.index, name=self._target_name
            )
        return data

    def _dummy_X(self, X, y):
        rX = self.algorithm_class.__name__ == "TemporalFusionTransformer"
        if rX and X is None:
            print(  # noqa T001
                "TemporalFusionTransformer requires X!\
                A constant dummy X with values all zero will be used!"
            )
            X = pd.DataFrame(data=np.zeros(len(y)), index=y.index)
        return X

    def _extend_y(self, y: pd.DataFrame, fh: ForecastingHorizon):
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


def _series_to_frame(data):
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
