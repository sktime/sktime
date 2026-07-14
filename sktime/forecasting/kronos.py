# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Kronos forecaster for ``sktime``."""

__author__ = ["shiyu-coder", "geetu040"]
# shiyu-coder for shiyu-coder/Kronos

__all__ = ["KronosForecaster"]

from copy import deepcopy

import numpy as np
import pandas as pd

from sktime.forecasting.foundation._base2 import BaseFoundationForecaster
from sktime.utils.singleton import _multiton


class KronosForecaster(BaseFoundationForecaster):
    """Kronos zero-shot forecaster for financial K-line/OHLC data.

    This forecaster wraps Kronos [1]_, a foundation model for financial market
    data [2]_, through the ``sktime`` forecasting interface. This implementation
    is inference-only and uses the upstream ``KronosPredictor`` preprocessing
    and autoregressive inference path internally.

    Parameters
    ----------
    model_path : str, default="NeoQuasar/Kronos-small"
        Hugging Face repository identifier or local path for the Kronos model.
        The default is the Kronos-small checkpoint [4]_. Other released
        checkpoints include Kronos-mini [3]_ and Kronos-base [5]_.
    tokenizer_path : str, default="NeoQuasar/Kronos-Tokenizer-base"
        Hugging Face repository identifier or local path for the Kronos tokenizer.
        The default is the Kronos-Tokenizer-base checkpoint [6]_. The released
        2k tokenizer is also available [7]_.
    device : str, default="cpu"
        Device used for model and tokenizer inference.
    columns : list of str or None, default=None
        Optional positional mapping from columns in ``y`` to Kronos internal
        columns. Positions map to ``"open"``, ``"high"``, ``"low"``,
        ``"close"``, ``"volume"``, and ``"amount"``. If provided, at least the
        first four OHLC columns are required; volume and amount are optional. If
        ``None``, literal open/high/low/close names are used when present,
        otherwise the first four numeric columns are used as OHLC. Literal
        volume/amount columns are used when present.
    freq : str or pandas offset, default="5min"
        Frequency used to synthesize timestamps for Kronos when the training
        index is not a ``pd.PeriodIndex`` or ``pd.DatetimeIndex``. The default
        is five minutes because Kronos is designed for financial K-line/OHLC
        data, where five-minute intraday bars are a common default granularity.
    start : str or pd.Timestamp, default="2000-01-01"
        Start timestamp used with ``freq`` to synthesize timestamps for Kronos
        when the training index is not datetime-like. The default date is
        arbitrary; it provides deterministic calendar fields for Kronos while
        keeping the synthetic timestamp path independent of the original index.
    clip : float, default=5.0
        Input normalization clipping value passed to ``KronosPredictor``.
    predict_kwargs : dict or None, default=None
        Additional keyword arguments passed directly to ``KronosPredictor.predict``.
        Examples are ``T``, ``top_k``, ``top_p``, ``sample_count``, and
        ``verbose``.
    deterministic : bool, default=False
        Whether predictions should reset the PyTorch random seed before
        autoregressive sampling.

    Notes
    -----
    ``KronosForecaster`` expects financial K-line style data. The input ``y``
    should contain OHLC columns, and may optionally contain volume and amount.
    For relative forecasting horizons with datetime-like indices, provide a
    sensible regular index/frequency so future timestamps line up with the data.
    If the calendar is irregular, pass an absolute forecasting horizon.

    References
    ----------
    .. [1] Kronos GitHub repository:
       https://github.com/shiyu-coder/Kronos
    .. [2] Lin, Z., Xia, Y., Liu, Z., Zhang, S., Wang, J., Yang, C., Dong, Q.,
       Liu, H., Jiang, H., Wang, S., Xiong, X., and Zhao, B. (2025).
       Kronos: A Foundation Model for the Language of Financial Markets.
       arXiv. https://arxiv.org/abs/2508.02739
    .. [3] Kronos-mini model card:
       https://huggingface.co/NeoQuasar/Kronos-mini
    .. [4] Kronos-small model card:
       https://huggingface.co/NeoQuasar/Kronos-small
    .. [5] Kronos-base model card:
       https://huggingface.co/NeoQuasar/Kronos-base
    .. [6] Kronos-Tokenizer-base model card:
       https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base
    .. [7] Kronos-Tokenizer-2k model card:
       https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k

    Examples
    --------
    >>> import pandas as pd
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.kronos import KronosForecaster
    >>> url = (
    ...     "https://raw.githubusercontent.com/shiyu-coder/Kronos/"
    ...     "refs/heads/master/tests/data/regression_input.csv"
    ... )
    >>> df = pd.read_csv(  # doctest: +SKIP
    ...     url,
    ...     parse_dates=["timestamps"],
    ...     index_col="timestamps",
    ... )
    >>> lookback, pred_len = 400, 120
    >>> y = df.iloc[:lookback]  # doctest: +SKIP
    >>> fh = ForecastingHorizon(  # doctest: +SKIP
    ...     df.index[lookback : lookback + pred_len],
    ...     is_relative=False,
    ... )
    >>> forecaster = KronosForecaster(  # doctest: +SKIP
    ...     model_path="NeoQuasar/Kronos-small",
    ...     tokenizer_path="NeoQuasar/Kronos-Tokenizer-base",
    ...     device="cpu",
    ...     deterministic=True,
    ...     predict_kwargs={
    ...         "T": 1.0,
    ...         "top_p": 0.9,
    ...         "sample_count": 1,
    ...         "verbose": True,
    ...     },
    ... )
    >>> y_pred = forecaster.fit(y).predict(fh=fh)  # doctest: +SKIP
    """

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": True,
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
        "tests:libs": ["sktime.libs.kronos"],
    }

    _kronos_columns = ["open", "high", "low", "close", "volume", "amount"]

    def __init__(
        self,
        model_path="NeoQuasar/Kronos-small",
        tokenizer_path="NeoQuasar/Kronos-Tokenizer-base",
        device="cpu",
        columns=None,
        freq="5min",
        start="2000-01-01",
        clip=5.0,
        predict_kwargs=None,
        deterministic=False,
    ):
        self.tokenizer_path = tokenizer_path
        self.columns = columns
        self.freq = freq
        self.start = start
        self.clip = clip
        self.predict_kwargs = predict_kwargs
        self.deterministic = deterministic
        super().__init__(
            model_path=model_path,
            device=device,
        )

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series to which to fit the forecaster.

            * if self.get_tag("capability:multivariate")==False:
              guaranteed to be univariate (e.g., single-column for DataFrame)
            * if self.get_tag("capability:multivariate")==True: no restrictions apply,
              the method should handle uni- and multivariate y appropriately

        fh : guaranteed to be ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series to fit to.

        Returns
        -------
        self : reference to self
        """
        self.context_ = y.copy()
        self.max_context_ = len(self.context_)

        self.column_mapping_ = self._resolve_column_mapping(y)
        self.output_columns_ = []
        for internal in self._kronos_columns:
            original = self.column_mapping_.get(internal)
            if (
                original is not None
                and original in y.columns
                and original not in self.output_columns_
            ):
                self.output_columns_.append(original)

        self.tokenizer_, self.model_ = self._load_kronos()

    def _predict(self, fh, X=None):
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
        X : sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        df = pd.DataFrame(index=self.context_.index)
        for internal, original in self.column_mapping_.items():
            df[internal] = self.context_[original].to_numpy()

        x_index = self.context_.index
        y_index = fh.to_absolute_index(self.cutoff)
        x_timestamp = self._to_timestamps(x_index)
        y_timestamp = self._to_timestamps(y_index)

        import torch

        from sktime.libs.kronos import KronosPredictor

        if self.deterministic:
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(42)

        predictor = KronosPredictor(
            self.model_,
            self.tokenizer_,
            device=self.device,
            max_context=self.max_context_,
            clip=self.clip,
        )
        predict_kwargs = deepcopy(self.predict_kwargs) if self.predict_kwargs else {}
        pred_df = predictor.predict(
            df=df,
            x_timestamp=x_timestamp,
            y_timestamp=y_timestamp,
            pred_len=len(y_timestamp),
            **predict_kwargs,
        )

        out = pd.DataFrame(index=y_index)
        for internal in self._kronos_columns:
            original = self.column_mapping_.get(internal)
            if (
                original in self.output_columns_
                and original not in out.columns
                and internal in pred_df.columns
            ):
                out[original] = pred_df[internal].to_numpy()
        out = out[self.output_columns_]

        return out

    def _to_timestamps(self, index):
        """Convert an sktime prediction index into timestamps for Kronos.

        Kronos does not use the index labels directly. The upstream predictor
        extracts calendar fields such as minute, hour, weekday, day, and month
        through pandas ``.dt`` accessors. That means integer and range indexes
        need a small translation layer before they can be passed in.

        Datetime-like indexes keep their own calendar. Other indexes are treated
        as step numbers on a synthetic regular calendar, using ``start`` and
        ``freq``. This keeps the fallback deterministic and lets non-time indexes
        still describe valid 5 minute bars by position.
        """
        if isinstance(index, pd.PeriodIndex):
            # Periods are time-aware, but Kronos expects timestamp-like values
            # for the `.dt` fields it builds internally.
            timestamp = index.to_timestamp()

        elif isinstance(index, pd.DatetimeIndex):
            # Already exactly what Kronos wants.
            timestamp = index

        else:
            # For plain integer/range indexes, use the labels as bar numbers.
            # This is better than just making `periods=len(index)` because the
            # forecast index can have gaps like [2, 5], and those gaps should
            # show up in the generated timestamps too.
            #
            # The date itself is not sacred. It is only an anchor so Kronos can
            # get minute/hour/day features; `freq` is what controls spacing.
            start = pd.Timestamp(self.start)
            offset = pd.tseries.frequencies.to_offset(self.freq)
            index_values = np.asarray(index, dtype=int)
            timestamp = [start + int(value) * offset for value in index_values]

        return pd.Series(timestamp)

    def _resolve_column_mapping(self, y):
        """Resolve user columns to Kronos' expected OHLCVA column names.

        The vendored predictor is fairly strict internally: it wants columns
        named ``open``, ``high``, ``low``, ``close``, and optionally ``volume``
        and ``amount``. sktime users may pass those exact names, pass a manual
        positional mapping, or just pass numeric columns in a generic DataFrame.

        This method keeps that translation in one place. It also records when
        we had to make a pragmatic fallback choice, so the rest of the forecaster
        can always build the Kronos input frame in the same internal format.
        """
        mapping = {}

        if self.columns is not None:
            # Explicit mapping wins. If the user provides it, assume they know
            # which columns are open/high/low/close and validate only the shape.

            if len(self.columns) < 4 or len(self.columns) > len(self._kronos_columns):
                raise ValueError(
                    "columns must contain 4 to 6 items, ordered as open, high, "
                    "low, close, volume, amount."
                )

            missing = [col for col in self.columns if col not in y.columns]
            if missing:
                raise ValueError(f"columns contains values missing from y: {missing}.")

            mapping.update(zip(self._kronos_columns, self.columns))

        elif all(col in y.columns for col in self._kronos_columns[:4]):
            # Best common case: the input already uses the names Kronos expects.
            for col in self._kronos_columns[:4]:
                mapping[col] = col

        else:
            # Last resort for sktime estimator checks and generic user data.
            # Kronos is an OHLC model, but the sktime interface should still be
            # able to run on a numeric DataFrame. Reusing columns is not ideal
            # market data semantics, but it is a deterministic fallback and lets
            # the model path exercise without inventing fake values elsewhere.

            numeric_cols = y.select_dtypes(include=[np.number]).columns.tolist()

            if len(numeric_cols) == 0:
                raise ValueError(
                    "KronosForecaster requires open/high/low/close columns, "
                    "a columns mapping, or at least one numeric column."
                )

            if len(numeric_cols) < 4:
                numeric_cols = [numeric_cols[i % len(numeric_cols)] for i in range(4)]

            for internal, original in zip(self._kronos_columns[:4], numeric_cols[:4]):
                mapping[internal] = original

        if self.columns is None:
            # Volume/amount are optional upstream. Only pass them through when
            # they are really present; the Kronos predictor fills missing ones.

            for col in self._kronos_columns[4:]:
                if col in y.columns:
                    mapping[col] = col

        return mapping

    def _load_kronos(self):
        """Load or retrieve cached Kronos tokenizer/model."""
        if hasattr(self, "model_") and hasattr(self, "tokenizer_"):
            return self.tokenizer_, self.model_

        tokenizer, model = _CachedKronos(
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
                "model_path": "NeoQuasar/Kronos-mini",
                "tokenizer_path": "NeoQuasar/Kronos-Tokenizer-2k",
                "deterministic": True,
            },
            {
                "model_path": "NeoQuasar/Kronos-mini",
                "tokenizer_path": "NeoQuasar/Kronos-Tokenizer-2k",
                "device": "cpu",
                "columns": None,
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
class _CachedKronos:
    """Multiton-backed cache wrapper for Kronos model/tokenizer pairs."""

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
        from sktime.libs.kronos import KronosTokenizer

        return KronosTokenizer.from_pretrained(self.tokenizer_path)

    def _load_model(self):
        from sktime.libs.kronos import Kronos

        return Kronos.from_pretrained(self.model_path)
