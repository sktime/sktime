"""Adapter for using MOIRAI Forecasters."""

from unittest.mock import patch

import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import BaseForecaster, _GlobalForecastingDeprecationMixin
from sktime.utils.singleton import _multiton

__author__ = ["gorold", "chenghaoliu89", "liu-jc", "benheid", "pranavvp16"]
# gorold, chenghaoliu89, liu-jc are from SalesforceAIResearch/uni2ts


class MOIRAIForecaster(_GlobalForecastingDeprecationMixin, BaseForecaster):
    """MOIRAI Forecasters.

    Parameters
    ----------
    checkpoint_path : str, default=None
        Path to the checkpoint of the model.
        Supported weights are available at [1]_.
    context_length : int, default=200
        Length of the context window, time points the model will take as input
        for inference.
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
    >>> from sktime.forecasting.moirai import MOIRAIForecaster
    >>> import pandas as pd
    >>> import numpy as np
    >>> morai_forecaster = MOIRAIForecaster(  # doctest: +SKIP
    ...     checkpoint_path="sktime/moirai-1.0-R-small"
    ... )
    >>> y = np.random.normal(0, 1, (30, 2))  # doctest: +SKIP
    >>> X = y * 2 + np.random.normal(0, 1, (30,1))  # doctest: +SKIP
    >>> index = pd.date_range("2020-01-01", periods=30, freq="D")  # doctest: +SKIP
    >>> y = pd.DataFrame(y, index=index)  # doctest: +SKIP
    >>> X = pd.DataFrame(X, columns=["x1", "x2"], index=index)  # doctest: +SKIP
    >>> morai_forecaster.fit(y, X=X)  # doctest: +SKIP
    MOIRAIForecaster(checkpoint_path='sktime/moirai-1.0-R-small')
    >>> X_test = pd.DataFrame(  # doctest: +SKIP
    ...     np.random.normal(0, 1, (10, 2)),
    ...     columns=["x1", "x2"],
    ...     index=pd.date_range("2020-01-31", periods=10, freq="D"),
    ... )
    >>> forecast = morai_forecaster.predict(fh=range(1, 11), X=X_test)  # doctest: +SKIP

    References
    ----------
    .. [1] https://huggingface.co/collections/sktime/moirai-variations-66ba3bc9f1dfeeafaed3b974
    .. [2] https://pypi.org/project/uni2ts/1.1.0/
    """

    _tags = {
        # packaging info
        # --------------
        "authors": ["gorold", "chenghaoliu89", "liu-jc", "benheid", "pranavvp16"],
        # gorold, chenghaoliu89, liu-jc are from SalesforceAIResearch/uni2ts
        "maintainers": ["pranavvp16"],
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
        "capability:unequal_length": False,
        "capability:pretrain": False,
        "property:randomness": "stochastic",
        # CI and test flags
        # -----------------
        "tests:vm": True,
    }

    def __init__(
        self,
        checkpoint_path: str,
        context_length=200,
        patch_size=32,
        num_samples=100,
        num_feat_dynamic_real=None,
        num_past_feat_dynamic_real=None,
        map_location=None,
        target_dim=2,
        broadcasting=False,
        deterministic=False,
        batch_size=32,
        use_source_package=False,
    ):
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
        super().__init__()

    def __dynamic_tags__(self):
        """Dynamic tag setter logic for setting tag values conditional on parameters.

        This method should be used for setting dynamic tags only.
        """
        if self.broadcasting:
            self.set_tags(
                **{
                    "y_inner_mtype": "pd.DataFrame",
                    "X_inner_mtype": "pd.DataFrame",
                    "capability:global_forecasting": False,
                }
            )

    def _get_moirai_kwargs(self):
        """Return model construction kwargs (excluding prediction_length).

        ``prediction_length`` is excluded from the cache key because it is
        updated at predict-time via ``self.model_.hparams.prediction_length``.

        The effective feature dimensions are read from the fitted attributes
        ``_feat_dynamic_real_dim_`` / ``_past_feat_dynamic_real_dim_`` when
        available (populated inside ``_fit``), falling back to the constructor
        parameters otherwise.  This ensures ``self.num_feat_dynamic_real``
        and ``self.num_past_feat_dynamic_real`` are never mutated.
        """
        _num_feat_dynamic_real = getattr(
            self,
            "_feat_dynamic_real_dim_",
            self.num_feat_dynamic_real if self.num_feat_dynamic_real is not None else 0,
        )
        _num_past_feat_dynamic_real = getattr(
            self,
            "_past_feat_dynamic_real_dim_",
            self.num_past_feat_dynamic_real
            if self.num_past_feat_dynamic_real is not None
            else 0,
        )
        return {
            "checkpoint_path": self.checkpoint_path,
            "context_length": self.context_length,
            "patch_size": self.patch_size,
            "num_samples": self.num_samples,
            "target_dim": self.target_dim,
            "feat_dynamic_real_dim": _num_feat_dynamic_real,
            "past_feat_dynamic_real_dim": _num_past_feat_dynamic_real,
            "map_location": self.map_location,
            "use_source_package": self.use_source_package,
        }

    def _get_moirai_key(self):
        """Return a unique cache key for the MOIRAI model."""
        return str(sorted(self._get_moirai_kwargs().items()))

    def _init_model(self, prediction_length=1):
        """Lazy-initialise the MOIRAI model, loading weights only once.

        If ``model_`` has already been set, the existing instance is returned without
        re-loading.

        Parameters
        ----------
        prediction_length : int, default=1
            Initial prediction length passed to the model constructor.
            This value can be overridden at predict-time.

        Returns
        -------
        model_ : MoiraiForecast
            The loaded (and possibly cached) MOIRAI model.
        """
        if not hasattr(self, "model_") or self.model_ is None:
            self.model_ = _CachedMoirai(
                key=self._get_moirai_key(),
                moirai_kwargs=self._get_moirai_kwargs(),
                prediction_length=prediction_length,
            ).load_from_checkpoint()
        return self.model_

    # ------------------------------------------------------------------
    # Pickle support
    # ------------------------------------------------------------------
    # ``model_`` is a PyTorch / uni2ts Lightning module whose internal
    # class references (e.g. ``uni2ts.distribution._base.DistrParamProj``)
    # cannot be resolved by Python's pickler when the vendored
    # ``sktime.libs.uni2ts`` package is used, because the classes are
    # registered under the original ``uni2ts.*`` namespace.
    #
    # Since the model is stateless (zero-shot) and already cached by the
    # multiton, we simply drop it from the pickle payload and restore it
    # lazily on the next ``predict()`` call.
    # ------------------------------------------------------------------

    def __getstate__(self):
        """Return picklable state with the non-serialisable model excluded."""
        state = self.__dict__.copy()
        state.pop("model_", None)
        return state

    def __setstate__(self, state):
        """Restore state; model_ will be reloaded lazily on next predict."""
        self.__dict__.update(state)
        self.model_ = None

    def _fit(self, y, X=None, fh=None):
        if fh is not None:
            prediction_length = max(fh.to_relative(self.cutoff))
        else:
            prediction_length = 1

        # Resolve effective feature dimensions without mutating hyper-parameters.
        # self.num_feat_dynamic_real / self.num_past_feat_dynamic_real stay as
        # the user set them (None means "infer from data").
        if self.num_feat_dynamic_real is None:
            self._feat_dynamic_real_dim_ = X.shape[1] if X is not None else 0
        else:
            self._feat_dynamic_real_dim_ = self.num_feat_dynamic_real

        if self.num_past_feat_dynamic_real is None:
            self._past_feat_dynamic_real_dim_ = 0
        else:
            self._past_feat_dynamic_real_dim_ = self.num_past_feat_dynamic_real

        # Lazy-init: load model on first access; reuse on subsequent fit() calls.
        self.model_ = self._init_model(prediction_length)
        self.model_.to(self.map_location)

    def _predict(self, fh, X=None):
        if fh is None:
            fh = self.fh
        fh = fh.to_relative(self.cutoff)

        # Re-load model from multiton cache if lost (e.g. after pickle round-trip).
        if not hasattr(self, "model_") or self.model_ is None:
            self._init_model()

        if self.deterministic:
            import torch

            torch.manual_seed(42)

        self.model_.hparams.prediction_length = max(fh._values)

        if min(fh._values) < 0:
            raise NotImplementedError(
                "The MORAI adapter is not supporting insample predictions."
            )

        _y = self._y.copy()
        _X = None
        if self._X is not None:
            _X = self._X.copy()

        # Zero shot case with X and fit data as context
        _use_fit_data_as_context = X is not None

        if isinstance(_y, pd.Series):
            target = [_y.name]
            _y, _is_converted_to_df = self._series_to_df(_y)
        else:
            target = _y.columns

        # Use local variables for predict-time state so that _predict
        # does not add new keys to __dict__ (sktime non-state-changing contract).
        _target_name = target
        _len_of_targets = len(target)

        target = [f"target_{i}" for i in range(_len_of_targets)]
        _y.columns = target

        future_length = 0
        feat_dynamic_real = None

        if _X is not None:
            feat_dynamic_real = [
                f"feat_dynamic_real_{i}" for i in range(self._X.shape[1])
            ]
            _X.columns = feat_dynamic_real

        pred_df = pd.concat([_y, _X], axis=1)
        _is_range_index = self.check_range_index(pred_df)
        _is_period_index = self.check_period_index(pred_df)

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
        # check whether the index is a PeriodIndex; store freq to restore later
        _period_freq = None
        if isinstance(pred_df.index, pd.PeriodIndex):
            time_idx = self.return_time_index(pred_df)
            _period_freq = time_idx.freq
            # Keep the natural DatetimeIndex freq from to_timestamp() so that
            # gluonts can infer the frequency. Setting freq=None breaks gluonts
            # internal infer_freq for short series (< 3 time points).
            pred_df.index = time_idx.to_timestamp()

        # For non-consecutive fh (e.g. fh=[2,5]) the future X only has rows at
        # those positions, leaving gaps that gluonts rejects as non-uniform.
        # Reindex to fill all intermediate steps and set future_length=max(fh).
        if _use_fit_data_as_context and not _is_range_index:
            if not isinstance(pred_df.index, pd.MultiIndex):
                # Use the RAW freq from the index for pd.date_range, NOT the
                # gluonts-mapped version. 'MS' (month-start) must stay 'MS';
                # mapping to 'M' would produce month-END dates that don't align.
                if hasattr(pred_df.index, "freq") and pred_df.index.freq is not None:
                    _raw_freq_str = pred_df.index.freqstr
                else:
                    _raw_freq_str = (
                        pd.infer_freq(pred_df.index[:3]) if len(pred_df) >= 3 else None
                    )
                if _raw_freq_str is not None:
                    full_idx = pd.date_range(
                        pred_df.index[0], pred_df.index[-1], freq=_raw_freq_str
                    )
                    if len(full_idx) > len(pred_df):
                        pred_df = pred_df.reindex(full_idx, fill_value=0)
                        future_length = int(max(fh._values))

        # Check if the index is a range index
        if _is_range_index:
            pred_df.index = self.handle_range_index(pred_df.index)

        _is_hierarchical = False
        if pred_df.index.nlevels >= 3:
            pred_df = self._convert_hierarchical_to_panel(pred_df)
            _is_hierarchical = True

        # Infer freq explicitly so gluonts doesn't have to; pd.concat/sort
        # strips the .freq attribute from DatetimeIndex, and with very few
        # data points pd.infer_freq returns None, crashing PandasDataset.
        _time_idx = self.return_time_index(pred_df)
        if _is_range_index:
            # range → daily DatetimeIndex created by handle_range_index
            _freq = "D"
        elif isinstance(_time_idx, pd.PeriodIndex):
            _freq = _time_idx.freqstr
        else:
            _freq = self.infer_freq(_time_idx)

        # Ensure the index is uniformly spaced before handing off to gluonts.
        # When X_test covers non-contiguous timepoints (e.g. fh=[2,4,6]) the
        # concat above can leave gaps; gluonts's PandasDataset raises
        # AssertionError if is_uniform(index) fails.
        if _freq is not None and not isinstance(pred_df.index, pd.MultiIndex):
            _tidx = pred_df.index
            _full = pd.date_range(_tidx[0], _tidx[-1], freq=_freq)
            if len(_full) > len(_tidx):
                pred_df = pred_df.reindex(_full, fill_value=0)

        # Recompute future_length from the actual pred_df after reindex so
        # that gluonts knows how many trailing rows are "future" context.
        # Using a stale len(X_test) after reindex causes gluonts to predict
        # fewer steps than prediction_length, making pred_out keys disappear.
        if _use_fit_data_as_context:
            _n_train = len(self.return_time_index(_y).unique())
            future_length = len(self.return_time_index(pred_df).unique()) - _n_train

        ds_test, df_config = self.create_pandas_dataset(
            pred_df, target, feat_dynamic_real, future_length, _target_name, _freq
        )

        predictor = self.model_.create_predictor(batch_size=self.batch_size)
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
        elif _period_freq is not None:
            # We converted pred_df from PeriodIndex to DatetimeIndex; gluonts
            # may return DatetimeIndex predictions. Convert back to PeriodIndex
            # so that pred_out (which has PeriodIndex) can index into predictions.
            if isinstance(predictions.index, pd.MultiIndex):
                dt_level = predictions.index.get_level_values(-1)
                if isinstance(dt_level, pd.DatetimeIndex):
                    predictions.index = predictions.index.set_levels(
                        levels=dt_level.to_period(_period_freq).unique(),
                        level=-1,
                    )
            elif isinstance(predictions.index, pd.DatetimeIndex):
                predictions.index = predictions.index.to_period(_period_freq)
        if _is_hierarchical:
            predictions = self._convert_panel_to_hierarchical(
                predictions, _y.index.names
            )

        pred_out = fh.get_expected_pred_idx(_y, cutoff=self.cutoff)

        if _is_range_index:
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

        # When the original data had a PeriodIndex we converted it to
        # DatetimeIndex before handing it to gluonts.  pred_out (derived from
        # _y) is still a PeriodIndex, so we must convert back before .loc[].
        if _is_period_index and not _is_range_index:
            _period_freq = self.return_time_index(_y).freqstr
            if isinstance(predictions.index, pd.MultiIndex):
                _last_level = predictions.index.get_level_values(-1)
                if isinstance(_last_level, pd.DatetimeIndex):
                    predictions.index = predictions.index.set_levels(
                        _last_level.to_period(_period_freq).unique(),
                        level=-1,
                    )
            elif isinstance(predictions.index, pd.DatetimeIndex):
                predictions.index = predictions.index.to_period(_period_freq)

        if _use_fit_data_as_context:
            # first_seen_index is X.index[0], which for MultiIndex data is a
            # tuple like ("item_A", Timestamp(...)). Extract the time component
            # (last element) so that pd.Period() / .loc[] receive a scalar.
            _fsi = first_seen_index
            if isinstance(_fsi, tuple):
                _fsi = _fsi[-1]
            if _period_freq is not None and not isinstance(_fsi, pd.Period):
                _fsi = pd.Period(_fsi, freq=_period_freq)
            elif _period_freq is None and isinstance(_fsi, pd.Period):
                _fsi = _fsi.to_timestamp()
            if isinstance(predictions.index, pd.MultiIndex):
                # Slice on the time level only
                time_vals = predictions.index.get_level_values(-1)
                predictions = predictions.loc[time_vals >= _fsi]
            else:
                predictions = predictions.loc[_fsi:]

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
        self,
        df,
        target,
        dynamic_features=None,
        forecast_horizon=0,
        original_target=None,
        freq=None,
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
        original_target : list, default=None
            Original (user-facing) target column names, stored in ``df_config``
            so that ``_get_prediction_df`` can rename columns back.  When
            ``None`` the renamed ``target`` list is used as-is.
        freq : str, default=None
            Pandas-compatible frequency string (e.g. ``"D"``, ``"M"``).
            When provided it is passed directly to ``PandasDataset`` /
            ``from_long_dataframe`` so that gluonts does not have to infer
            it from the index (which can fail when the index has no ``.freq``
            attribute after a ``pd.concat`` / ``sort_index`` call).

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
            "target": original_target if original_target is not None else target,
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

            _from_long_kwargs = dict(
                target=target,
                feat_dynamic_real=dynamic_features,
                item_id=item_id,
                future_length=forecast_horizon,
            )
            if freq is not None:
                _from_long_kwargs["freq"] = freq
            dataset = PandasDataset.from_long_dataframe(df, **_from_long_kwargs)
        else:
            _pd_kwargs = dict(
                target=target,
                feat_dynamic_real=dynamic_features,
                future_length=forecast_horizon,
            )
            if freq is not None:
                _pd_kwargs["freq"] = freq
            dataset = PandasDataset(df, **_pd_kwargs)

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
        # Mapping from pandas DatetimeIndex freq aliases (produced by
        # PeriodIndex.to_timestamp()) to gluonts-compatible freq strings.
        _offset_to_period = {
            "MS": "M",
            "ME": "M",
            "QS": "Q",
            "QE": "Q",
            "QS-OCT": "Q",
            "YS": "Y",
            "AS": "Y",
            "YE": "Y",
            "AE": "Y",
        }
        if isinstance(index, pd.PeriodIndex):
            return index.freqstr
        # Prefer the freq attribute already set on the index (e.g. by to_timestamp())
        if hasattr(index, "freq") and index.freq is not None:
            freq_str = index.freqstr
            return _offset_to_period.get(freq_str, freq_str)
        # Fall back to inferring from the first 3 values
        freq = pd.infer_freq(index[:3]) if len(index) >= 3 else None
        if freq is not None:
            return _offset_to_period.get(freq, freq)
        return freq

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


@_multiton
class _CachedMoirai:
    """Cached MOIRAI model to ensure only one instance exists in memory.

    MOIRAI is a zero-shot model and immutable, hence there are no side effects
    of sharing the same instance across multiple uses.  The multiton pattern
    (keyed on model configuration) prevents redundant weight downloads when
    ``fit()`` is called multiple times with different data.

    ``prediction_length`` is *not* part of the cache key because it is updated
    at inference time via ``self.model_.hparams.prediction_length`` in
    ``MOIRAIForecaster._predict()``.
    """

    def __init__(self, key, moirai_kwargs, prediction_length):
        self.key = key
        self.moirai_kwargs = moirai_kwargs
        self.prediction_length = prediction_length
        self.model = None

    def load_from_checkpoint(self):
        """Return the cached model, loading weights on first call."""
        if self.model is not None:
            return self.model

        # Guard against incompatible hf_xet (e.g., PyO3 ABI mismatch when
        # hf_xet was compiled for an older CPython than the current runtime).
        # huggingface_hub reads HF_HUB_DISABLE_XET from constants.py at
        # import time, and file_download.py accesses it as
        # `constants.HF_HUB_DISABLE_XET` at call time. We must therefore
        # patch huggingface_hub.constants directly (not file_download) so
        # that _download_to_tmp_and_move skips xet_get for this session.
        import os

        try:
            import hf_xet  # noqa: F401
        except Exception:
            os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
            if _check_soft_dependencies("huggingface_hub", severity="none"):
                import huggingface_hub.constants as _hf_constants

                _hf_constants.HF_HUB_DISABLE_XET = True

        kwargs = self.moirai_kwargs
        model_kwargs = {
            "prediction_length": self.prediction_length,
            "context_length": kwargs["context_length"],
            "patch_size": kwargs["patch_size"],
            "num_samples": kwargs["num_samples"],
            "target_dim": kwargs["target_dim"],
            "feat_dynamic_real_dim": kwargs["feat_dynamic_real_dim"],
            "past_feat_dynamic_real_dim": kwargs["past_feat_dynamic_real_dim"],
        }

        if kwargs["use_source_package"]:
            if _check_soft_dependencies("uni2ts", severity="none"):
                from uni2ts.model.moirai import MoiraiForecast, MoiraiModule

                if kwargs["checkpoint_path"].startswith("Salesforce"):
                    model_kwargs["module"] = MoiraiModule.from_pretrained(
                        kwargs["checkpoint_path"]
                    )
                    self.model = MoiraiForecast(**model_kwargs)
                else:
                    from huggingface_hub import hf_hub_download

                    model_kwargs["checkpoint_path"] = hf_hub_download(
                        repo_id=kwargs["checkpoint_path"], filename="model.ckpt"
                    )
                    # weights_only=False: PyTorch>=2.6 changed the default to True,
                    # but MOIRAI checkpoints contain trusted uni2ts globals that
                    # cannot be loaded with weights_only=True.
                    self.model = MoiraiForecast.load_from_checkpoint(
                        **model_kwargs, weights_only=False
                    )
        else:
            # Use the sktime-vendored uni2ts package with sys.modules patched
            # so that ``import uni2ts`` inside MoiraiForecast resolves to
            # ``sktime.libs.uni2ts``.
            import sktime
            from sktime.libs.uni2ts.forecast import MoiraiForecast

            with patch.dict("sys.modules", {"uni2ts": sktime.libs.uni2ts}):
                if kwargs["checkpoint_path"].startswith("Salesforce"):
                    from sktime.libs.uni2ts.moirai_module import MoiraiModule

                    model_kwargs["module"] = MoiraiModule.from_pretrained(
                        kwargs["checkpoint_path"]
                    )
                    self.model = MoiraiForecast(**model_kwargs)
                else:
                    from huggingface_hub import hf_hub_download

                    model_kwargs["checkpoint_path"] = hf_hub_download(
                        repo_id=kwargs["checkpoint_path"], filename="model.ckpt"
                    )
                    # weights_only=False: PyTorch>=2.6 changed the default to True,
                    # but MOIRAI checkpoints contain trusted uni2ts globals that
                    # cannot be loaded with weights_only=True.
                    self.model = MoiraiForecast.load_from_checkpoint(
                        **model_kwargs, weights_only=False
                    )

        return self.model
