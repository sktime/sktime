# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""WindFM forecaster for ``sktime``."""

__author__ = ["shiyu-coder", "geetu040"]
# shiyu-coder for shiyu-coder/WindFM

__all__ = ["WindFMForecaster"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.base import BaseForecaster
from sktime.utils.singleton import _multiton
from sktime.utils.warnings import warn


class WindFMForecaster(BaseForecaster):
    """WindFM zero-shot forecaster for wind power data.

    This forecaster wraps WindFM [1]_, a foundation model for wind power
    forecasting [2]_, through the ``sktime`` forecasting interface. This
    implementation is inference-only and uses the upstream ``WindFMPredictor``
    preprocessing and autoregressive inference path internally.

    Parameters
    ----------
    model_path : str or None, default="NeoQuasar/WindFM"
        Hugging Face repository identifier or local path for the WindFM model.
        The default is the WindFM checkpoint [3]_. Other released
        checkpoints include WindFM-robust [4]_. If ``None`` together with
        ``tokenizer_path=None``, a lightweight randomly initialized WindFM model
        and tokenizer are used, primarily for estimator testing.
    tokenizer_path : str or None, default="NeoQuasar/WindFM-Tokenizer"
        Hugging Face repository identifier or local path for the WindFM tokenizer.
        The default is the WindFM-Tokenizer checkpoint [5]_. The released
        WindFM-Tokenizer-robust is also available [6]_. If ``None`` together
        with ``model_path=None``, a lightweight randomly initialized WindFM model
        and tokenizer are used, primarily for estimator testing.
    device : str, default="cpu"
        Device used for model and tokenizer inference.
    columns : list of str or None, default=None
        Optional mapping from columns in ``X`` to WindFM weather covariates.
        If provided, it must contain five entries ordered as ``"wind_speed"``,
        ``"wind_direction"``, ``"density"``, ``"temperature"``, and
        ``"pressure"``. If ``None``, ``X`` must contain the literal WindFM
        covariate names.
    freq : str or pandas offset, default="1h"
        Frequency used to synthesize timestamps for WindFM when the training
        index is not a ``pd.PeriodIndex`` or ``pd.DatetimeIndex``. The default
        is one hour, matching the example data distributed by WindFM.
    start : str or pd.Timestamp, default="2000-01-01"
        Start timestamp used with ``freq`` to synthesize timestamps for WindFM
        when the training index is not datetime-like. The default date is
        arbitrary; it provides deterministic calendar fields for WindFM while
        keeping the synthetic timestamp path independent of the original index.
    clip : float, default=5.0
        Input normalization clipping value passed to ``WindFMPredictor``.
    predict_kwargs : dict or None, default=None
        Additional keyword arguments passed directly to ``WindFMPredictor.predict``.
        Examples are ``T``, ``top_k``, ``top_p``, ``sample_count``, and
        ``verbose``.
    deterministic : bool, default=False
        Whether predictions should reset the PyTorch random seed before
        autoregressive sampling.

    Notes
    -----
    ``WindFMForecaster`` expects ``y`` to contain the target wind power series
    and ``X`` to contain the historical weather covariates required by WindFM:
    wind speed, wind direction, density, temperature, and pressure. WindFM
    generates sample paths internally; this wrapper returns the median across
    those samples as a point forecast.

    WindFM was trained with UTC timestamps. If the input index is not
    datetime-like, synthetic timestamps are generated from ``start`` and
    ``freq``.

    References
    ----------
    .. [1] WindFM GitHub repository:
       https://github.com/shiyu-coder/WindFM
    .. [2] Hang Fan, Yu Shi, Zongliang Fu, Shuo Chen, Wei Wei, Wei Xu, Jian Li (2025).
       WindFM: An Open-Source Foundation Model for Zero-Shot Wind Power Forecasting.
       arXiv. https://arxiv.org/abs/2509.06311
    .. [3] WindFM model card:
       https://huggingface.co/NeoQuasar/WindFM
    .. [4] WindFM-robust model card:
       https://huggingface.co/NeoQuasar/WindFM-robust
    .. [5] WindFM-Tokenizer model card:
       https://huggingface.co/NeoQuasar/WindFM-Tokenizer
    .. [6] WindFM-Tokenizer-robust model card:
       https://huggingface.co/NeoQuasar/WindFM-Tokenizer-robust

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.windfm import WindFMForecaster
    >>> url = (
    ...     "https://raw.githubusercontent.com/shiyu-coder/WindFM/"
    ...     "refs/heads/master/tests/data/regression_input.csv"
    ... )
    >>> df = pd.read_csv(  # doctest: +SKIP
    ...     url,
    ...     parse_dates=["timestamps"],
    ...     index_col="timestamps",
    ... )
    >>> lookback, pred_len = 400, 120
    >>> y = df["power"].iloc[:lookback]  # doctest: +SKIP
    >>> X = df[  # doctest: +SKIP
    ...     ["wind_speed", "wind_direction", "density", "temperature", "pressure"]
    ... ].iloc[:lookback]
    >>> fh = ForecastingHorizon(  # doctest: +SKIP
    ...     df.index[lookback : lookback + pred_len],
    ...     is_relative=False,
    ... )
    >>> forecaster = WindFMForecaster(  # doctest: +SKIP
    ...     model_path="NeoQuasar/WindFM",
    ...     tokenizer_path="NeoQuasar/WindFM-Tokenizer",
    ...     device="cpu",
    ...     deterministic=True,
    ...     predict_kwargs={
    ...         "T": 1.0,
    ...         "top_p": 0.9,
    ...         "sample_count": 1,
    ...         "verbose": True,
    ...     },
    ... )
    >>> y_pred = forecaster.fit(y, X=X).predict(fh=fh)  # doctest: +SKIP
    """

    _tags = {
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "X-y-must-have-same-index": True,
        "capability:multivariate": False,
        "capability:exogenous": True,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pretrain": False,
        "authors": ["shiyu-coder", "geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": [
            "torch",
            "einops",
            "huggingface_hub",
            "tqdm",
            "safetensors",
        ],
        "tests:vm": True,
        "tests:libs": ["sktime.libs.windfm"],
    }

    _target_col = "power"
    _covariate_cols = [
        "wind_speed",
        "wind_direction",
        "density",
        "temperature",
        "pressure",
    ]

    def __init__(
        self,
        model_path="NeoQuasar/WindFM",
        tokenizer_path="NeoQuasar/WindFM-Tokenizer",
        device="cpu",
        columns=None,
        freq="1h",
        start="2000-01-01",
        clip=5.0,
        predict_kwargs=None,
        deterministic=False,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.columns = columns
        self.freq = freq
        self.start = start
        self.clip = clip
        self.predict_kwargs = predict_kwargs
        self.deterministic = deterministic

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to target power and historical weather covariates."""
        if X is None:
            warn(
                "WindFMForecaster requires weather covariates in X for meaningful "
                "WindFM forecasts. Since X was not provided, the target series y "
                "will be reused as placeholder covariates.",
                obj=self,
            )
            X = self._make_fallback_X(y)

        self.y_name_ = y.name
        self.y_context_ = y.copy()
        self.X_context_ = X.copy()
        self.max_context_ = len(self.y_context_)
        self.column_mapping_ = self._resolve_column_mapping(self.X_context_)
        self.tokenizer_, self.model_ = self._load_windfm()
        return self

    def _predict(self, fh, X=None):
        """Forecast target power at future horizon."""
        pred_df, y_index = self._predict_samples(fh)
        point_pred = pred_df.median(axis=1).to_numpy()
        return pd.Series(point_pred, index=y_index, name=self.y_name_)

    def _predict_quantiles(self, fh, X=None, alpha=None):
        """Forecast target power quantiles from WindFM sample paths."""
        pred_df, y_index = self._predict_samples(fh)

        quantiles = np.quantile(pred_df.to_numpy(), q=alpha, axis=1).T
        columns = pd.MultiIndex.from_product([self._get_varnames(), alpha])
        return pd.DataFrame(quantiles, index=y_index, columns=columns)

    def _predict_samples(self, fh):
        """Generate WindFM sample paths for a future horizon."""
        df = self._make_windfm_frame()

        x_index = self.y_context_.index
        y_index = fh.to_absolute_index(self.cutoff)
        x_timestamp = self._to_timestamps(x_index)
        y_timestamp = self._to_timestamps(y_index)

        import torch

        if self.deterministic:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

        predictor = self._make_predictor()
        predict_kwargs = deepcopy(self.predict_kwargs) if self.predict_kwargs else {}
        pred_df = predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=len(y_timestamp),
            **predict_kwargs,
        )

        return pred_df, y_index

    def _make_predictor(self):
        """Instantiate the upstream WindFM predictor."""
        from sktime.libs.windfm import WindFMPredictor

        return WindFMPredictor(
            self.model_,
            self.tokenizer_,
            device=self.device,
            max_context=self.max_context_,
            clip=self.clip,
        )

    def _to_timestamps(self, index):
        """Convert an sktime prediction index into timestamps for WindFM.

        WindFM does not use the index labels directly. The upstream predictor
        extracts calendar fields such as minute, hour, weekday, day, and month
        through pandas ``.dt`` accessors. That means integer and range indexes
        need a small translation layer before they can be passed in.

        Datetime-like indexes keep their own calendar. Other indexes are treated
        as step numbers on a synthetic regular calendar, using ``start`` and
        ``freq``. This keeps the fallback deterministic and lets non-time indexes
        still describe valid 5 minute bars by position.
        """
        if isinstance(index, pd.PeriodIndex):
            # Periods are time-aware, but WindFM expects timestamp-like values
            # for the `.dt` fields it builds internally.
            timestamp = index.to_timestamp()

        elif isinstance(index, pd.DatetimeIndex):
            # Already exactly what WindFM wants.
            timestamp = index

        else:
            # For plain integer/range indexes, use the labels as bar numbers.
            # This is better than just making `periods=len(index)` because the
            # forecast index can have gaps like [2, 5], and those gaps should
            # show up in the generated timestamps too.
            #
            # The date itself is not sacred. It is only an anchor so WindFM can
            # get minute/hour/day features; `freq` is what controls spacing.
            start = pd.Timestamp(self.start)
            offset = pd.tseries.frequencies.to_offset(self.freq)
            index_values = np.asarray(index, dtype=int)
            timestamp = [start + int(value) * offset for value in index_values]

        return pd.Series(timestamp)

    def _make_windfm_frame(self):
        """Build the upstream WindFM input frame from y and X."""
        df = pd.DataFrame(index=self.y_context_.index)
        for internal, original in self.column_mapping_.items():
            df[internal] = self.X_context_[original].to_numpy()
        df[self._target_col] = self.y_context_.to_numpy()
        feature_cols = (
            self._covariate_cols[:2] + [self._target_col] + self._covariate_cols[2:]
        )
        return df[feature_cols]

    def _resolve_column_mapping(self, X):
        """Resolve X columns to WindFM's weather covariate names."""
        if self.columns is None:
            mapping = {col: col for col in self._covariate_cols}
        else:
            if not isinstance(self.columns, list):
                raise ValueError("columns must be a list of five column names or None.")
            if len(self.columns) != len(self._covariate_cols):
                raise ValueError(
                    "columns must contain five items ordered as wind_speed, "
                    "wind_direction, density, temperature, pressure."
                )
            mapping = dict(zip(self._covariate_cols, self.columns))

        missing = [col for col in mapping.values() if col not in X.columns]
        if missing and self._uses_random_windfm():
            columns = list(X.columns)
            mapping = {
                internal: columns[i % len(columns)]
                for i, internal in enumerate(self._covariate_cols)
            }
            missing = []
        if missing:
            raise ValueError(f"columns contains values missing from X: {missing}.")
        return mapping

    def _load_windfm(self):
        """Load or retrieve cached WindFM tokenizer/model."""
        if hasattr(self, "model_") and hasattr(self, "tokenizer_"):
            return self.tokenizer_, self.model_
        if self._uses_random_windfm():
            return self._init_random_windfm()
        if self.model_path is None or self.tokenizer_path is None:
            raise ValueError(
                "model_path and tokenizer_path must either both be None, or both "
                "identify WindFM Hugging Face repositories or local paths."
            )

        tokenizer, model = _CachedWindFM(
            key=self._get_unique_key(),
            model_path=self.model_path,
            tokenizer_path=self.tokenizer_path,
            device=self.device,
        ).load()

        return tokenizer, model

    def _get_unique_key(self):
        """Build cache key for the multiton model loader."""
        key = {
            "model_path": self.model_path,
            "tokenizer_path": self.tokenizer_path,
            "device": self.device,
        }
        return str(sorted(key.items()))

    def _uses_random_windfm(self):
        """Return whether this instance should use random WindFM weights."""
        return self.model_path is None and self.tokenizer_path is None

    def _make_fallback_X(self, y):
        """Create placeholder covariates from y when X is omitted."""
        return pd.DataFrame(
            {col: y.to_numpy() for col in self._covariate_cols},
            index=y.index,
        )

    def _init_random_windfm(self):
        """Initialize a small random WindFM model/tokenizer pair for tests."""
        from sktime.libs.windfm import WindFM, WindFMTokenizer

        tokenizer = WindFMTokenizer(
            d_in=6,
            d_model=8,
            n_heads=1,
            ff_dim=16,
            n_enc_layers=1,
            n_dec_layers=1,
            ffn_dropout_p=0.0,
            attn_dropout_p=0.0,
            resid_dropout_p=0.0,
            s1_bits=2,
            s2_bits=2,
            beta=0.0,
            gamma0=0.0,
            gamma=0.0,
            zeta=0.0,
            group_size=4,
        )
        model = WindFM(
            s1_bits=2,
            s2_bits=2,
            n_layers=1,
            d_model=8,
            n_heads=1,
            ff_dim=16,
            ffn_dropout_p=0.0,
            attn_dropout_p=0.0,
            resid_dropout_p=0.0,
            token_dropout_p=0.0,
            learn_te=False,
        )
        return tokenizer, model

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
        return [
            {
                "model_path": None,
                "tokenizer_path": None,
                "columns": [0, 1, 0, 1, 0],
                "deterministic": True,
            },
            {
                "model_path": None,
                "tokenizer_path": None,
                "device": "cpu",
                "columns": [0, 1, 0, 1, 0],
                "clip": 5.0,
                "deterministic": True,
                "predict_kwargs": {
                    "T": 0.8,
                    "sample_count": 1,
                    "top_k": 1,
                    "top_p": 1.0,
                    "verbose": False,
                },
            },
        ]


@_multiton
class _CachedWindFM:
    """Multiton-backed cache wrapper for WindFM model/tokenizer pairs."""

    def __init__(
        self,
        key,
        model_path,
        tokenizer_path,
        device,
    ):
        self.key = key
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.device = device
        self.tokenizer_ = None
        self.model_ = None

    def load(self):
        """Load tokenizer and model if needed and return cached pair."""
        if self.tokenizer_ is not None and self.model_ is not None:
            return self.tokenizer_, self.model_

        self.tokenizer_ = self._load_tokenizer()
        self.model_ = self._load_model()
        self.tokenizer_ = self.tokenizer_.to(self.device).eval()
        self.model_ = self.model_.to(self.device).eval()
        return self.tokenizer_, self.model_

    def _load_tokenizer(self):
        from sktime.libs.windfm import WindFMTokenizer

        return WindFMTokenizer.from_pretrained(self.tokenizer_path)

    def _load_model(self):
        from sktime.libs.windfm import WindFM

        return WindFM.from_pretrained(self.model_path)
