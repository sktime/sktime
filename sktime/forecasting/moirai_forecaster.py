"""Adapter for using MOIRAI Forecasters."""

from unittest.mock import patch

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("lightning", severity="none"):
    import sktime.libs.uni2ts

from sktime.forecasting.base import _BaseGlobalForecaster

if _check_soft_dependencies("huggingface-hub", severity="none"):
    from huggingface_hub import hf_hub_download


__author__ = ["gorold", "chenghaoliu89", "liu-jc", "benheid", "pranavvp16"]
# gorold, chenghaoliu89, liu-jc are from SalesforceAIResearch/uni2ts


class MOIRAIForecaster(_BaseGlobalForecaster):
    """
    Adapter for using MOIRAI Forecasters.

    Parameters
    ----------
    checkpoint_path : str, default=None
        Path to the checkpoint of the model. Supported weights are available at [1]_.
    context_length : int, default=200
        Length of the context window, time points the model with take input as infernce.
    patch_size : int, default=32
        Time steps to perform patching with.
    num_samples : int, default=100
        Number of samples to draw.
    map_location : str, default=None
        Hardware to use for the model.
    target_dim : int, default=2
        Dimension of the target.
    deterministic : bool, default=False
        Whether to use a deterministic model.
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
        package ``uni2ts.models.moirai``. This is useful if you
        want to bypass the local version of the package or when working in an
        environment where the latest updates from the source package are needed.
        If False, the model and configuration will be loaded from the local
        version of package maintained in sktime.
        To install the source package, follow the instructions here [2]_.

    Examples
    --------
    >>> from sktime.forecasting.moirai_forecaster import MOIRAIForecaster
    >>> import pandas as pd
    >>> import numpy as np
    >>> morai_forecaster = MOIRAIForecaster(
    ...     checkpoint_path=f"sktime/moirai-1.0-R-small"
    ... )
    >>> y = np.random.normal(0, 1, (30, 2))
    >>> X = y * 2 + np.random.normal(0, 1, (30,1))
    >>> index = pd.date_range("2020-01-01", periods=30, freq="D")
    >>> y = pd.DataFrame(y, index=index)
    >>> X = pd.DataFrame(X, columns=["x1", "x2"], index=index)
    >>> morai_forecaster.fit(y, X=X)
    >>> X_test = pd.DataFrame(np.random.normal(0, 1, (10, 2)),
    ...                      columns=["x1", "x2"],
    ...                      index=pd.date_range("2020-01-31", periods=10, freq="D"),
    ... )
    >>> forecast = morai_forecaster.predict(fh=range(1, 11), X=X_test)

    References
    ----------
    .. [1] https://huggingface.co/collections/sktime/moirai-variations-66ba3bc9f1dfeeafaed3b974
    .. [2] https://pypi.org/project/uni2ts/1.1.0/
    """

    _tags = {
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": False,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "capability:pred_int": False,
        "python_dependencies": [
            "gluonts",
            "torch",
            "einops",
            "huggingface-hub",
            "lightning",
            "hydra-core",
        ],
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
        "authors": ["gorold", "chenghaoliu89", "liu-jc", "benheid", "pranavvp16"],
        # gorold, chenghaoliu89, liu-jc are from SalesforceAIResearch/uni2ts
        "maintainers": ["pranavvp16"],
    }

    def __init__(
        self,
        checkpoint_path: str,
        context_length=200,
        patch_size=32,
        num_samples=100,
        num_feat_dynamic_real=0,
        num_past_feat_dynamic_real=0,
        map_location=None,
        target_dim=2,
        broadcasting=False,
        deterministic=False,
        batch_size=32,
        use_source_package=False,
    ):
        super().__init__()
        self.checkpoint_path = checkpoint_path
        self.context_length = context_length
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.num_past_feat_dynamic_real = num_past_feat_dynamic_real
        self.map_location = map_location
        self.target_dim = target_dim
        self.broadcasting = broadcasting
        self.deterministic = deterministic
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

    # Apply a patch for redirecting imports to sktime.libs.uni2ts
    if _check_soft_dependencies(["lightning", "huggingface-hub"], severity="none"):
        from sktime.libs.uni2ts.forecast import MoiraiForecast

        @patch.dict("sys.modules", {"uni2ts": sktime.libs.uni2ts})
        def _instantiate_patched_model(self, model_kwargs):
            """Instantiate the model from the vendor package."""
            from sktime.libs.uni2ts.forecast import MoiraiForecast

            if self.checkpoint_path.startswith("Salesforce"):
                from sktime.libs.uni2ts.moirai_module import MoiraiModule

                model_kwargs["module"] = MoiraiModule.from_pretrained(
                    self.checkpoint_path
                )
                return MoiraiForecast(**model_kwargs)
            else:
                model_kwargs["checkpoint_path"] = hf_hub_download(
                    repo_id=self.checkpoint_path, filename="model.ckpt"
                )
                return MoiraiForecast.load_from_checkpoint(**model_kwargs)

    def _fit(self, y, X, fh):
        if fh is not None:
            prediction_length = max(fh.to_relative(self.cutoff))
        else:
            prediction_length = 1

        if self.num_feat_dynamic_real is None and X is not None:
            self.num_feat_dynamic_real = X.shape[1]
        if self.num_past_feat_dynamic_real is None:
            self.num_past_feat_dynamic_real = 0

        model_kwargs = {
            "prediction_length": prediction_length,
            "context_length": self.context_length,
            "patch_size": self.patch_size,
            "num_samples": self.num_samples,
            "target_dim": self.target_dim,
            "feat_dynamic_real_dim": self.num_feat_dynamic_real,
            "past_feat_dynamic_real_dim": self.num_past_feat_dynamic_real,
        }

        # Load model from source package
        if self.use_source_package:
            if _check_soft_dependencies("uni2ts", severity="none"):
                from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

                if self.checkpoint_path.startswith("Salesforce"):
                    model_kwargs["module"] = MoiraiModule.from_pretrained(
                        self.checkpoint_path
                    )
                    self.model = MoiraiForecast(**model_kwargs)
                else:
                    model_kwargs["checkpoint_path"] = hf_hub_download(
                        repo_id=self.checkpoint_path, filename="model.ckpt"
                    )
                    self.model = MoiraiForecast.load_from_checkpoint(**model_kwargs)
                    self.model.to(self.map_location)
        # Load model from sktime
        else:
            self.model = self._instantiate_patched_model(model_kwargs)
            self.model.to(self.map_location)

    def _predict(self, fh, y=None, X=None):
        if self.deterministic:
            import torch

            torch.manual_seed(42)

        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        self.model.hparams.prediction_length = max(fh._values)

        if min(fh._values) < 0:
            raise NotImplementedError(
                "The MORAI adapter is not supporting insample predictions."
            )

        _y = self._y.copy()
        _X = None
        if self._X is not None:
            _X = self._X.copy()

        # Zero shot case with X and fit data as context
        _use_fit_data_as_context = False
        if X is not None and y is None:
            _use_fit_data_as_context = True

        # Override to data in fit as new timeseries is passed
        elif y is not None:
            _y = y.copy()
            if X is not None:
                _X = X.copy()

        if isinstance(_y, pd.Series):
            target = [_y.name]
            _y, _is_converted_to_df = self._series_to_df(_y)
        else:
            target = _y.columns

        # Store the original index and target name
        self._target_name = target
        self._len_of_targets = len(target)

        target = [f"target_{i}" for i in range(self._len_of_targets)]
        _y.columns = target

        future_length = 0
        feat_dynamic_real = None

        if _X is not None:
            feat_dynamic_real = [
                f"feat_dynamic_real_{i}" for i in range(self._X.shape[1])
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
        # check whether the index is a PeriodIndex
        if isinstance(pred_df.index, pd.PeriodIndex):
            time_idx = self.return_time_index(pred_df)
            pred_df.index = time_idx.to_timestamp()
            pred_df.index.freq = None

        # Check if the index is a range index
        if self._is_range_index:
            pred_df.index = self.handle_range_index(pred_df.index)

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
                # Convert str type to int
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
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """
        return [
            {
                "deterministic": True,
                "checkpoint_path": "sktime/moirai-1.0-R-small",
            },
            {
                "deterministic": True,
                "checkpoint_path": "Salesforce/moirai-1.0-R-small",
            },
        ]

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

    def create_pandas_dataset(
        self, df, target, dynamic_features=None, forecast_horizon=0
    ):
        """Create a gluonts PandasDataset from the input data.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        target : str
            Target column name.
        dynamic_features : list, default=None
            List of dynamic features.
        forecast_horizon : int, default=0
            Forecast horizon.

        Returns
        -------
        dataset : PandasDataset
            Pandas dataset.
        df_config : dict
            Configuration of the input data.

        """
        if _check_soft_dependencies("gluonts", severity="none"):
            from gluonts.dataset.pandas import PandasDataset

        # Add original target to config
        df_config = {
            "target": self._target_name,
        }

        # PandasDataset expects non-multiindex dataframe with item_id
        # and timepoints
        if isinstance(df.index, pd.MultiIndex):
            if None in df.index.names:
                df.index.names = ["item_id", "timepoints"]
            item_id = df.index.names[0]
            df_config["item_id"] = item_id
            timepoints = df.index.names[-1]
            df_config["timepoints"] = timepoints

            # Reset index to create a non-multiindex dataframe
            df = df.reset_index()
            df.set_index(timepoints, inplace=True)

            dataset = PandasDataset.from_long_dataframe(
                df,
                target=target,
                feat_dynamic_real=dynamic_features,
                item_id=item_id,
                future_length=forecast_horizon,
            )
        else:
            dataset = PandasDataset(
                df,
                target=target,
                feat_dynamic_real=dynamic_features,
                future_length=forecast_horizon,
            )

        return dataset, df_config

    # def _extend_df(self, df, _y, X=None):
    #     """Extend the input dataframe up to the timepoints that need to be predicted.
    #
    #     Parameters
    #     ----------
    #     df : pd.DataFrame
    #         Input data that needs to be extended
    #     X : pd.DataFrame, default=None
    #         Assumes that X has future timepoints and is
    #         concatenated to the input data,
    #         if X is present in the input, but None here the values of X are assumed
    #         to be 0 in future timepoints that need to be predicted.
    #     is_range_index : bool, default=False
    #         If True, the index is a range index.
    #     is_period_index : bool, default=False
    #         If True, the index is a period index.
    #
    #     Returns
    #     -------
    #     pd.DataFrame
    #         Extended dataframe with future timepoints.
    #     """
    #     index = self.return_time_index(df)
    #
    #     # Extend the index to the future timepoints
    #     # respective to index last seen
    #
    #     if self._is_range_index:
    #         pred_index = pd.RangeIndex(
    #             self.cutoff[0] + 1, self.cutoff[0] + max(self.fh._values)
    #         )
    #     elif self._is_period_index:
    #         pred_index = pd.period_range(
    #             self.cutoff[0],
    #             periods=max(self.fh._values) + 1,
    #             freq=index.freq,
    #         )[1:]
    #     else:
    #         pred_index = pd.date_range(
    #             self.cutoff[0],
    #             periods=max(self.fh._values) + 1,
    #             freq=self.infer_freq(index),
    #         )[1:]
    #
    #     if isinstance(df.index, pd.MultiIndex):
    #         # Works for any number of levels in the MultiIndex
    #         index_levels = [
    #             df.index.get_level_values(i).unique()
    #             for i in range(df.index.nlevels - 1)
    #         ]
    #         index_levels.append(pred_index)
    #         new_index = pd.MultiIndex.from_product(index_levels, names=df.index.names)
    #     else:
    #         new_index = pred_index
    #
    #     df_y = pd.DataFrame(columns=_y.columns, index=new_index)
    #     df_y.fillna(0, inplace=True)
    #     pred_df = pd.concat([df_y, X], axis=1)
    #     extended_df = pd.concat([df, pred_df])
    #     extended_df = extended_df.sort_index()
    #     extended_df.fillna(0, inplace=True)
    #
    #     return extended_df, df_y

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
        """
        Convert RangeIndex to Dummy DatetimeIndex.

        As gluonts PandasDataset expects a DatetimeIndex.
        """
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

    def _get_future_length(self, X):
        """Get the future length."""
        if isinstance(X.index, pd.MultiIndex):
            return len(X.index.get_level_values(-1).unique())
        else:
            return len(X)
