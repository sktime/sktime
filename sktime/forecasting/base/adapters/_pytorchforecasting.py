# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements adapter for pytorch-forecasting models."""
import abc
import functools
import typing
from copy import deepcopy
from typing import Any, Dict, Optional

import pandas
from pandas.api.types import is_numeric_dtype

from sktime.forecasting.base import BaseGlobalForecaster, ForecastingHorizon

__all__ = ["_PytorchForecastingAdapter"]
__author__ = ["XinyuWu"]


class _PytorchForecastingAdapter(BaseGlobalForecaster):
    """Base adapter class for pytorch-forecasting models."""

    _tags = {
        # packaging info
        # --------------
        "authors": ["XinyuWu"],
        "maintainers": ["XinyuWu"],
        "python_version": ">=3.8",
        "python_dependencies": ["pytorch_forecasting"],
        # estimator type
        # --------------
        "y_inner_mtype": ["pd-multiindex", "pd_multiindex_hier", "pd.Series"],
        "X_inner_mtype": ["pd-multiindex", "pd_multiindex_hier"],
        "scitype:y": "univariate",
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "handles-missing-data": False,
        "capability:insample": False,
    }

    def __init__(
        self: "_PytorchForecastingAdapter",
        model_params: Optional[Dict[str, Any]] = None,
        dataset_params: Optional[Dict[str, Any]] = None,
        train_to_dataloader_params: Optional[Dict[str, Any]] = None,
        validation_to_dataloader_params: Optional[Dict[str, Any]] = None,
        trainer_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.model_params = model_params
        self.dataset_params = dataset_params
        self.trainer_params = trainer_params
        self.train_to_dataloader_params = train_to_dataloader_params
        self.validation_to_dataloader_params = validation_to_dataloader_params

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
        self._model_params = _none_check(self.model_params, {})
        algorithm_instance = self.algorithm_class.from_dataset(
            data,
            **self.algorithm_parameters,
            **self._model_params,
        )
        self._trainer_params = _none_check(self.trainer_params, {})
        import lightning.pytorch as pl

        traner_instance = pl.Trainer(**self._trainer_params)
        return algorithm_instance, traner_instance

    def _fit(
        self: "_PytorchForecastingAdapter",
        y: pandas.DataFrame,
        X: typing.Optional[pandas.DataFrame],
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

        Raises
        ------
        ValueError
            When ``freq="auto"`` and cannot be interpreted from ``ForecastingHorizon``
        """
        self._dataset_params = _none_check(self.dataset_params, {})
        self._max_prediction_length = fh.to_relative()[-1]
        # convert series to frame
        if isinstance(y, pandas.Series):
            _y = deepcopy(y).to_frame()
        else:
            _y = deepcopy(y)
        # store the target column name and index names(probably [None])
        self._target_name = _y.columns[-1]
        self._index_names = _y.index.names
        self._index_len = len(self._index_names)
        # store X, y column names (probably None or not str type)
        if X is not None:
            self._X_columns = X.columns.tolist()
        # convert data to pytorch-forecasting datasets
        training, validation = self._Xy_to_dataset(
            deepcopy(X), _y, self._dataset_params, self._max_prediction_length
        )
        self._forecaster, self._trainer = self._instantiate_model(training)
        self._train_to_dataloader_params = {"train": True}
        self._train_to_dataloader_params.update(
            _none_check(self.train_to_dataloader_params, {})
        )
        self._validation_to_dataloader_params = {"train": False}
        self._validation_to_dataloader_params.update(
            _none_check(self.validation_to_dataloader_params, {})
        )
        self._trainer.fit(
            self._forecaster,
            train_dataloaders=training.to_dataloader(
                **self._train_to_dataloader_params
            ),
            val_dataloaders=validation.to_dataloader(
                **self._validation_to_dataloader_params
            ),
        )
        return self

    def _predict(
        self: "_PytorchForecastingAdapter",
        fh: typing.Optional[ForecastingHorizon],
        X: typing.Optional[pandas.DataFrame],
        y: typing.Optional[pandas.DataFrame],
    ) -> pandas.Series:
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
        y : sktime time series object, optional (default=None)
            Historical values of the time series that should be predicted.

        Returns
        -------
        y_pred : sktime time series object
            guaranteed to have a single column/variable
            Point predictions
        """
        # convert series to frame
        if isinstance(y, pandas.Series):
            _y = deepcopy(y).to_frame()
            self._convert_to_series = True
        else:
            _y = deepcopy(y)
            self._convert_to_series = False
        # convert data to pytorch-forecasting datasets
        training, validation = self._Xy_to_dataset(
            deepcopy(X), _y, self._dataset_params, self._max_prediction_length
        )
        # load model from checkpoint
        best_model_path = self._trainer.checkpoint_callback.best_model_path
        best_model = self.algorithm_class.load_from_checkpoint(best_model_path)
        predictions = best_model.predict(
            validation.to_dataloader(**self._validation_to_dataloader_params),
            return_x=True,
            return_index=True,
            return_decoder_lengths=True,
        )
        # convert pytorch-forecasting predictions to dataframe
        output = self._predictions_to_dataframe(
            predictions, self._max_prediction_length
        )

        return output

    def _Xy_to_dataset(
        self,
        X: pandas.DataFrame,
        y: pandas.DataFrame,
        dataset_params: Dict[str, Any],
        max_prediction_length,
    ):
        from pytorch_forecasting.data import TimeSeriesDataSet

        # X, y must have same index or X is None
        assert X is None or (X.index == y.index).all()
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
            time_varying_known_reals = [
                c for c in X.columns if is_numeric_dtype(X[c].dtype)
            ]
            data = X.join(y, on=X.index.names)
        else:
            time_varying_known_reals = []
            data = deepcopy(y)
        # add int time_idx as pytorch-forecasting requires
        if self._index_len > 1:
            time_idx = (
                data.groupby(by=self._new_index_names[0:-1]).cumcount().to_frame()
            )
        else:
            time_idx = pandas.DataFrame(range(0, len(data)))
            time_idx.index = data.index
        time_idx.rename(columns={0: "_auto_time_idx"}, inplace=True)
        data = pandas.concat([data, time_idx], axis=1)
        # reset multi index to normal columns
        data = data.reset_index(level=list(range(self._index_len)))
        training_cutoff = data["_auto_time_idx"].max() - max_prediction_length
        # add a constant column as group id if data only contains only one timeseries
        if self._index_len == 1:
            group_id = pandas.DataFrame([0] * len(data))
            group_id.index = data.index
            group_id.rename(columns={0: "_auto_group_id"}, inplace=True)
            data = pandas.concat([group_id, data], axis=1)
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

    def _predictions_to_dataframe(self, predictions, max_prediction_length):
        # output is the predictions
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
        # add the target column at the end
        data[self._target_name] = output.flatten()
        # correct the time_idx after repeating
        # assume the time_idx column is continuous integers
        for i in range(output.shape[0]):
            start_idx = i * max_prediction_length
            start_time = data.loc[start_idx, time_idx]
            data.loc[
                start_idx : start_idx + max_prediction_length - 1, time_idx
            ] = list(range(start_time, start_time + max_prediction_length))

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
        # set target name back to original input in fit
        data.columns = [self._target_name]
        # convert back to pd.series if needed
        if self._convert_to_series:
            data = pandas.Series(
                data=data[self._target_name], index=data.index, name=self._target_name
            )
        return data


def _none_check(value, default):
    return value if value is not None else default
