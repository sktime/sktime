"""Adapter for using MOIRAI 2.0 Forecasters."""

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import _BaseGlobalForecaster

__author__ = ["gorold", "chenghaoliu89", "liu-jc", "priyanshuharshbodhi1"]


class Moirai2Forecaster(_BaseGlobalForecaster):
    """
    Adapter for using MOIRAI 2.0 Forecasters.

    MOIRAI 2.0 is a decoder-only universal time series foundation model that
    uses quantile predictions instead of distributional sampling. It outputs
    predictions at 9 quantile levels (0.1 through 0.9), with point forecasts
    taken from the median (0.5) quantile.

    Parameters
    ----------
    checkpoint_path : str
        Path to the checkpoint of the model. Supported weights are available at [1]_.
    context_length : int, default=200
        Length of the context window, time points the model will take as input.
    map_location : str, default=None
        Hardware to use for the model.
    target_dim : int, default=2
        Dimension of the target.
    batch_size : int, default=32
        Number of samples in each batch of inference.
    broadcasting : bool, default=False
        if True, multiindex data input will be broadcasted to single series.
        For each single series, one copy of this forecaster will try to
        fit and predict on it. The broadcasting is happening inside automatically,
        from the outerside api perspective, the input and output are the same,
        only one multiindex output from ``predict``
    use_source_package : bool, default=False
        If True, the model and configuration will be loaded directly from the source
        package ``uni2ts.models.moirai2``. This is useful if you
        want to bypass the local version of the package or when working in an
        environment where the latest updates from the source package are needed.
        If False, the model and configuration will be loaded from the local
        version of package maintained in sktime.
        To install the source package, follow the instructions here [2]_.

    Examples
    --------
    >>> from sktime.forecasting.moirai2_forecaster import Moirai2Forecaster
    >>> import pandas as pd
    >>> import numpy as np
    >>> forecaster = Moirai2Forecaster(  # doctest: +SKIP
    ...     checkpoint_path="Salesforce/moirai-2.0-R-small"
    ... )
    >>> y = np.random.normal(0, 1, (30, 2))
    >>> index = pd.date_range("2020-01-01", periods=30, freq="D")
    >>> y = pd.DataFrame(y, index=index)  # doctest: +SKIP
    >>> forecaster.fit(y)  # doctest: +SKIP
    Moirai2Forecaster(checkpoint_path='Salesforce/moirai-2.0-R-small')

    References
    ----------
    .. [1] https://huggingface.co/collections/Salesforce/moirai-r-models-65c8d3a94c51428c300e0742
    .. [2] https://pypi.org/project/uni2ts/
    """

    _tags = {
        # packaging info
        # --------------
        "authors": [
            "gorold",
            "chenghaoliu89",
            "liu-jc",
            "priyanshuharshbodhi1",
        ],
        # gorold, chenghaoliu89, liu-jc are from SalesforceAIResearch/uni2ts
        "maintainers": ["priyanshuharshbodhi1"],
        "python_version": "<3.14",
        "python_dependencies": [
            "gluonts",
            "torch",
            "einops",
            "huggingface_hub",
            "hf-xet",
            "lightning",
            "hydra-core",
            "safetensors",
        ],
        # estimator type
        # --------------
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "capability:missing_values": False,
        "capability:pred_int": False,
        "X_inner_mtype": ["pd.DataFrame", "pd-multiindex", "pd_multiindex_hier"],
        "y_inner_mtype": [
            "pd.Series",
            "pd.DataFrame",
            "pd-multiindex",
            "pd_multiindex_hier",
        ],
        "capability:insample": False,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        checkpoint_path: str,
        context_length=200,
        num_feat_dynamic_real=None,
        num_past_feat_dynamic_real=None,
        map_location=None,
        target_dim=2,
        broadcasting=False,
        batch_size=32,
        use_source_package=False,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.context_length = context_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_past_feat_dynamic_real = num_past_feat_dynamic_real
        self.map_location = map_location
        self.target_dim = target_dim
        self.broadcasting = broadcasting
        self.batch_size = batch_size
        self.use_source_package = use_source_package

        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.DataFrame",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

    def _instantiate_patched_model(self, model_kwargs):
        """Instantiate the model from the local sktime uni2ts copy."""
        import sys

        import sktime.libs.uni2ts as _uni2ts_mod

        sys.modules.setdefault("uni2ts", _uni2ts_mod)
        from sktime.libs.uni2ts.moirai2_forecast import Moirai2Forecast

        if self.checkpoint_path.startswith("Salesforce"):
            from sktime.libs.uni2ts.moirai2_module import Moirai2Module

            model_kwargs["module"] = Moirai2Module.from_pretrained(
                self.checkpoint_path
            )
            return Moirai2Forecast(**model_kwargs)
        else:
            from huggingface_hub import hf_hub_download

            model_kwargs["checkpoint_path"] = hf_hub_download(
                repo_id=self.checkpoint_path, filename="model.ckpt"
            )
            return Moirai2Forecast.load_from_checkpoint(**model_kwargs)

    def _fit(self, y, X, fh):
        if fh is not None:
            prediction_length = max(fh.to_relative(self.cutoff))
        else:
            prediction_length = 1

        if self.num_feat_dynamic_real is None:
            if X is not None:
                self.num_feat_dynamic_real = X.shape[1]
            else:
                self.num_feat_dynamic_real = 0

        if self.num_past_feat_dynamic_real is None:
            self.num_past_feat_dynamic_real = 0

        if isinstance(y, pd.DataFrame):
            target_dim = y.shape[1]
        else:
            target_dim = 1

        model_kwargs = {
            "prediction_length": prediction_length,
            "context_length": self.context_length,
            "target_dim": target_dim,
            "feat_dynamic_real_dim": self.num_feat_dynamic_real,
            "past_feat_dynamic_real_dim": self.num_past_feat_dynamic_real,
        }

        if self.use_source_package:
            if _check_soft_dependencies("uni2ts", severity="none"):
                from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

                if self.checkpoint_path.startswith("Salesforce"):
                    model_kwargs["module"] = Moirai2Module.from_pretrained(
                        self.checkpoint_path
                    )
                    self.model = Moirai2Forecast(**model_kwargs)
                else:
                    from huggingface_hub import hf_hub_download

                    model_kwargs["checkpoint_path"] = hf_hub_download(
                        repo_id=self.checkpoint_path, filename="model.ckpt"
                    )
                    self.model = Moirai2Forecast.load_from_checkpoint(**model_kwargs)
                    self.model.to(self.map_location)
        else:
            self.model = self._instantiate_patched_model(model_kwargs)
            self.model.to(self.map_location)

    def _predict(self, fh, y=None, X=None):
        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        self.model.hparams.prediction_length = max(fh._values)

        if min(fh._values) < 0:
            raise NotImplementedError(
                "The Moirai2 adapter is not supporting insample predictions."
            )

        _y = self._y.copy()
        _X = None
        if self._X is not None:
            _X = self._X.copy()

        _use_fit_data_as_context = False
        if X is not None and y is None:
            _use_fit_data_as_context = True
        elif y is not None:
            _y = y.copy()
            if X is not None:
                _X = X.copy()

        if isinstance(_y, pd.Series):
            target = [_y.name]
            _y, _is_converted_to_df = self._series_to_df(_y)
        else:
            target = _y.columns

        self._target_name = target
        self._len_of_targets = len(target)

        target = [f"target_{i}" for i in range(self._len_of_targets)]
        _y.columns = target

        future_length = 0
        feat_dynamic_real = None

        if _X is not None:
            feat_dynamic_real = [
                f"feat_dynamic_real_{i}" for i in range(_X.shape[1])
            ]
            _X.columns = feat_dynamic_real

        pred_df = pd.concat([_y, _X], axis=1)
        self._is_range_index = self.check_range_index(pred_df)
        self._is_period_index = self.check_period_index(pred_df)

        if _use_fit_data_as_context:
            future_length = self._get_future_length(X)
            first_seen_index = X.index[0]
            X_to_extend = X.copy()
            X_to_extend.columns = feat_dynamic_real
            _X = pd.concat([_X, X_to_extend]).sort_index()
            pred_df = pd.concat([_y, _X], axis=1).sort_index()
            pred_df.fillna(0, inplace=True)
        else:
            if _X is not None:
                future_length = len(_X.index.get_level_values(-1).unique()) - len(
                    _y.index.get_level_values(-1).unique()
                )
            else:
                future_length = 0

        if isinstance(pred_df.index, pd.PeriodIndex):
            time_idx = self.return_time_index(pred_df)
            pred_df.index = time_idx.to_timestamp()
            pred_df.index.freq = None

        if self._is_range_index:
            pred_df.index = self.handle_range_index(pred_df.index)

        if _use_fit_data_as_context and not self._is_range_index:
            if not isinstance(pred_df.index, pd.MultiIndex):
                raw_freq = pd.infer_freq(pred_df.index[:3])
                if raw_freq is not None:
                    full_idx = pd.date_range(
                        pred_df.index[0], pred_df.index[-1], freq=raw_freq
                    )
                    if len(full_idx) > len(pred_df):
                        pred_df = pred_df.reindex(full_idx, fill_value=0)
                        future_length = int(max(fh._values))

        _is_hierarchical = False
        if pred_df.index.nlevels >= 3:
            pred_df = self._convert_hierarchical_to_panel(pred_df)
            _is_hierarchical = True

        ds_test, df_config = self.create_pandas_dataset(
            pred_df, target, feat_dynamic_real, future_length
        )

        predictor = self.model.create_predictor(batch_size=self.batch_size)
        forecasts = predictor.predict(ds_test)
        forecast_it = iter(forecasts)
        predictions = self._get_prediction_df(forecast_it, df_config)
        if isinstance(_y.index.get_level_values(-1), pd.DatetimeIndex):
            if isinstance(predictions.index, pd.MultiIndex):
                predictions.index = predictions.index.set_levels(
                    levels=predictions.index.get_level_values(-1)
                    .to_timestamp()
                    .unique(),
                    level=-1,
                )
            else:
                predictions.index = predictions.index.to_timestamp()
        if _is_hierarchical:
            predictions = self._convert_panel_to_hierarchical(
                predictions, _y.index.names
            )

        pred_out = fh.get_expected_pred_idx(_y, cutoff=self.cutoff)

        if self._is_range_index:
            timepoints = self.return_time_index(predictions)
            timepoints = timepoints.to_timestamp()
            timepoints = (timepoints - pd.Timestamp("2010-01-01")).map(
                lambda x: x.days
            ) + self.return_time_index(_y)[0]
            if isinstance(predictions.index, pd.MultiIndex):
                predictions.index = predictions.index.set_levels(
                    levels=timepoints.unique(), level=-1
                )
                predictions.index = predictions.index.map(lambda x: (int(x[0]), x[1]))
            else:
                predictions.index = timepoints

        if _use_fit_data_as_context:
            predictions = predictions.loc[first_seen_index:]

        predictions = predictions.loc[pred_out]
        predictions.index = pred_out
        return predictions

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        return [
            {
                "checkpoint_path": "Salesforce/moirai-2.0-R-small",
            },
            {
                "checkpoint_path": "Salesforce/moirai-2.0-R-small",
                "context_length": 100,
            },
        ]

    def _get_prediction_df(self, forecast_iter, df_config):
        def handle_series_prediction(forecast, target):
            pred = forecast.quantile_ts("0.5")
            if target[0] is not None:
                return pred.rename(target[0])
            else:
                return pred

        def handle_panel_predictions(forecasts_it, df_config):
            panels = []
            for forecast in forecasts_it:
                df = forecast.quantile_ts("0.5").reset_index()
                df.columns = [df_config["timepoints"], df_config["target"][0]]
                df[df_config["item_id"]] = forecast.item_id
                df.set_index(
                    [df_config["item_id"], df_config["timepoints"]], inplace=True
                )
                panels.append(df)
            return pd.concat(panels)

        forecasts = list(forecast_iter)

        if forecasts[0].item_id is None:
            return handle_series_prediction(forecasts[0], df_config["target"])
        else:
            return handle_panel_predictions(forecasts, df_config)

    def create_pandas_dataset(
        self, df, target, dynamic_features=None, forecast_horizon=0
    ):
        """Create a gluonts PandasDataset from the input data."""
        if _check_soft_dependencies("gluonts", severity="none"):
            from gluonts.dataset.pandas import PandasDataset

        df_config = {
            "target": self._target_name,
        }

        if isinstance(df.index, pd.MultiIndex):
            if None in df.index.names:
                df.index.names = ["item_id", "timepoints"]
            item_id = df.index.names[0]
            df_config["item_id"] = item_id
            timepoints = df.index.names[-1]
            df_config["timepoints"] = timepoints

            df = df.reset_index()
            df.set_index(timepoints, inplace=True)

            ds_target = target[0] if len(target) == 1 else target
            dataset = PandasDataset.from_long_dataframe(
                df,
                target=ds_target,
                feat_dynamic_real=dynamic_features,
                item_id=item_id,
                future_length=forecast_horizon,
            )
        else:
            ds_target = target[0] if len(target) == 1 else target
            dataset = PandasDataset(
                df,
                target=ds_target,
                feat_dynamic_real=dynamic_features,
                future_length=forecast_horizon,
            )

        return dataset, df_config

    def infer_freq(self, index):
        """Infer frequency of the index."""
        if isinstance(index, pd.PeriodIndex):
            return index.freq
        return pd.infer_freq(index[:3])

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

    def check_period_index(self, df):
        """Check if the index is a PeriodIndex."""
        timepoints = self.return_time_index(df)
        if isinstance(timepoints, pd.PeriodIndex):
            return True
        return False

    def handle_range_index(self, index):
        """Convert RangeIndex to dummy DatetimeIndex."""
        start_date = "2010-01-01"
        if isinstance(index, pd.MultiIndex):
            n_periods = index.levels[-1].size
            datetime_index = pd.date_range(
                start=start_date, periods=n_periods, freq="D"
            )
            new_index = index.set_levels(datetime_index, level=-1)
        else:
            n_periods = index.size
            new_index = pd.date_range(start=start_date, periods=n_periods, freq="D")
        return new_index

    def _series_to_df(self, y):
        """Convert series to DataFrame."""
        is_converted = False
        if isinstance(y, pd.Series):
            y = y.to_frame()
            is_converted = True
        return y, is_converted

    def _convert_hierarchical_to_panel(self, df):
        data = df.copy()
        flattened_index = [("*".join(map(str, x[:-1])), x[-1]) for x in data.index]
        data.index = pd.MultiIndex.from_tuples(
            flattened_index, names=["Flattened_Level", data.index.names[-1]]
        )
        return data

    def _convert_panel_to_hierarchical(self, df, original_index_names=None):
        if original_index_names is None:
            original_index_names = df.index.names

        data = df.reset_index()

        split_levels = data["Flattened_Level"].str.split("*", expand=True)
        split_levels.columns = original_index_names[:-1]
        index_names = split_levels.columns.tolist()

        data_converted = pd.concat(
            [split_levels, data.drop(columns=["Flattened_Level"])], axis=1
        )

        last_index_name = (
            original_index_names[-1]
            if original_index_names[-1] is not None
            else "timepoints"
        )

        data_converted = data_converted.set_index(index_names + [last_index_name])

        return data_converted

    def _get_future_length(self, X):
        """Get the future length."""
        if isinstance(X.index, pd.MultiIndex):
            return len(X.index.get_level_values(-1).unique())
        else:
            return len(X)
