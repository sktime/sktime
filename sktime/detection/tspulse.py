# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interface for TSPulse anomaly detection via granite-tsfm."""

__author__ = ["Faakhir30"]
__all__ = ["TSPulseAnomalyDetector"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _safe_import
from sktime.utils.singleton import _multiton

torch = _safe_import("torch")

TIMESTAMP_COL = "__TSPULSE_TIMESTAMP_COL__"

_DEFAULT_MODEL_CONFIG = {
    "ignore_mismatched_sizes": True,
}


class TSPulseAnomalyDetector(BaseDetector):
    """Anomaly detector wrapping IBM TSPulse via granite-tsfm.

    Loads a pretrained ``TSPulseForReconstruction`` checkpoint and scores each
    time point with the Hugging Face ``TimeSeriesAnomalyDetectionPipeline`` for
    zero-shot detection.

    ``DatetimeIndex`` on ``X`` is required: it is copied into an internal
    timestamp column. If ``X`` has no ``DatetimeIndex``, a synthetic daily
    timestamp column is added.

    Implementation adapted from [1].

    Parameters
    ----------
    model_path : str, default="ibm-granite/granite-timeseries-tspulse-r1"
        Hugging Face model id or local path.
    revision : str, default="main"
        Model revision on the Hugging Face Hub.
    mask_type : str, default="user"
        Patch-masking strategy during reconstruction. ``"user"`` applies masking
        from an optional ``past_observed_mask`` (the default for anomaly
        scoring). Other supported values include ``"block"``, ``"hybrid"``,
        ``"var_hybrid"``, and ``"random"``.
    prediction_mode : str or list of str, optional
        Score type(s) computed on each context window before they are merged.
        Supported values:

        - ``"time"``: mean squared error between the input and the model's
          time-domain reconstruction over the last ``aggregation_length`` points
          of the context window.
        - ``"fft"``: same error using the reconstruction from the frequency
          (FFT) branch.
        - ``"forecast"``: mean squared error between the next observed value
          and the model's one-step forecast.

        When ``None`` (default), uses ``["time", "fft"]``. A single string is
        wrapped in a one-element list. With several types, each produces its
        own score sequence; see ``smoothing_length`` and ``aggr_function``.
    aggregation_length : int, default=64
        Length of the context suffix used for patchwise stitched reconstruction
        when scoring with ``"time"`` or ``"fft"``. Also controls boundary padding
        when scores are aligned back to the full series length.
    aggr_function : str, default="max"
        How to merge score sequences from different entries in
        ``prediction_mode`` at each time point. One of ``"max"``, ``"min"``, or
        ``"mean"``.
    smoothing_length : int, default=8
        Moving-average window applied to each score sequence listed in
        ``prediction_mode`` after boundary alignment, before ``aggr_function``
        merges them. One-step ``"forecast"`` scores skip this smoothing unless
        ``predictive_score_smoothing`` is ``True``.
    least_significant_scale : float, default=0.01
        Value in ``(0, 1)``. Sets a variance-based floor on raw errors before
        scores are rescaled: deviations smaller than this fraction of the
        squared differences in the (standardized) input are treated as
        insignificant.
    least_significant_score : float, default=0.1
        Minimum scale factor applied when normalizing significant errors to the
        ``[0, 1]`` score range returned by the pipeline.
    batch_size : int, default=128
        Batch size for pipeline inference.
    predictive_score_smoothing : bool, default=False
        If ``True``, apply ``smoothing_length`` smoothing to ``"forecast"``
        scores as well; if ``False``, forecast scores are left unsmoothed.
    anomaly_threshold : float, optional, default=0.6
        Scores strictly above this value are flagged as anomalies in
        ``predict``. If ``None``, ``anomaly_percentile`` is used instead.
    anomaly_percentile : float, optional, default=None
        Percentile of the score vector on ``X`` used as the detection threshold
        when ``anomaly_threshold`` is ``None``. Exactly one of
        ``anomaly_threshold`` and ``anomaly_percentile`` must be set.
    config : dict, optional, default=None
        Extra keyword arguments forwarded to
        ``TSPulseForReconstruction.from_pretrained``.

        Any key you supply overrides the built-in default for that key, *except*
        for ``num_input_channels``, which is always set from the number of
        columns in ``X`` at ``fit`` time.

        If ``config`` is ``None``, the following default override is applied:

        - ``ignore_mismatched_sizes=True``:
          allows loading when channel or head shapes differ from the checkpoint

        Other commonly useful overrides (see ``TSPulseConfig`` in granite-tsfm):

        - ``context_length``: history length per window (checkpoint default often
          512 for r1 models)
        - ``patch_length`` / ``patch_stride``: patch size and stride for the encoder
        - ``decoder_mode``: channel mixing in the decoder (``"mix_channel"`` or
          ``"common_channel"``)
        - ``scaling``: input normalization (``"revin"``, ``"mean"``, ``"std"``, or
          ``None``)
        - ``mask_ratio``: fraction of patches masked when ``mask_type`` is not
          ``"user"``
        - ``fft_time_consistent_masking``: if ``True``, masked series is used for
          the FFT branch during training-style masking
        - ``reconstruction_type``: ``"patchwise"`` or ``"full"`` reconstruction
        - ``prediction_length``: horizon for the forecast head when using
          ``prediction_mode="forecast"``

    References
    ----------
    .. [1] https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/tspulse_anomaly_detection.ipynb
    .. [2] https://arxiv.org/abs/2505.13033

    Examples
    --------
    >>> from sktime.detection.tspulse import TSPulseAnomalyDetector
    >>> import pandas as pd
    >>> import numpy as np
    >>> idx = pd.date_range("2020-01-01", periods=200, freq="D")
    >>> X = pd.DataFrame(np.random.randn(200, 1), index=idx, columns=["value"])
    >>> detector = TSPulseAnomalyDetector()  # doctest: +SKIP
    >>> detector.fit(X)  # doctest: +SKIP
    >>> detector.predict(X)  # doctest: +SKIP
    >>> detector.predict_scores(X)  # doctest: +SKIP
    """

    _tags = {
        "authors": [
            "wgifford",
            "ajati",
            "subodh2702",
            "vijaye12ibm",
            "summukhe",
            "Tomoya Sakai",
            "Pankaj Dayama",
            "Jayant Kalagnanam",
            "Faakhir30",
        ],
        "maintainers": ["Faakhir30"],
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:missing_values": False,
        "fit_is_empty": False,
        "python_version": ">=3.11",
        "python_dependencies": [
            "granite-tsfm>=0.3.5",
            "torch",
            "transformers",
            "accelerate",
        ],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str = "ibm-granite/granite-timeseries-tspulse-r1",
        revision: str = "main",
        mask_type: str = "user",
        prediction_mode=None,
        aggregation_length: int = 64,
        aggr_function: str = "max",
        smoothing_length: int = 8,
        least_significant_scale: float = 0.01,
        least_significant_score: float = 0.1,
        batch_size: int = 128,
        predictive_score_smoothing: bool = False,
        anomaly_threshold: float | None = 0.6,
        anomaly_percentile: float | None = None,
        config: dict | None = None,
    ):
        self.model_path = model_path
        self.revision = revision
        self.mask_type = mask_type
        self.prediction_mode = prediction_mode
        self.aggregation_length = aggregation_length
        self.aggr_function = aggr_function
        self.smoothing_length = smoothing_length
        self.least_significant_scale = least_significant_scale
        self.least_significant_score = least_significant_score
        self.batch_size = batch_size
        self.predictive_score_smoothing = predictive_score_smoothing
        self.anomaly_threshold = anomaly_threshold
        self.anomaly_percentile = anomaly_percentile
        self.config = config

        super().__init__()

    def __post_init__(self):
        """Post-initialization logic for TSPulseAnomalyDetector."""
        from tsfm_public.toolkit.ad_helpers import AnomalyScoreMethods

        self._config = {} if self.config is None else self.config.copy()
        if self.prediction_mode is None:
            self._prediction_mode = [
                AnomalyScoreMethods.TIME_RECONSTRUCTION.value,
                AnomalyScoreMethods.FREQUENCY_RECONSTRUCTION.value,
            ]
        elif isinstance(self.prediction_mode, str):
            self._prediction_mode = [self.prediction_mode]
        else:
            self._prediction_mode = list(self.prediction_mode)

        self._device = "cpu"
        if _check_soft_dependencies("torch", severity="none"):
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"

        if self.anomaly_threshold is None and self.anomaly_percentile is None:
            raise ValueError(
                "Error in initializing TSPulseAnomalyDetector, "
                "either anomaly_threshold or anomaly_percentile must be set."
            )

    def _fit(self, X, y=None):
        if TIMESTAMP_COL in X.columns:
            raise ValueError(
                f'Column name "{TIMESTAMP_COL}" is reserved for internal use; '
                "rename the column in X."
            )
        self._target_columns = list(X.columns)
        self._num_input_channels = X.shape[1]
        self._model = _CachedTSPulseModel(
            key=self._get_unique_model_key(),
            detector=self,
        ).load_from_checkpoint()
        self._pipeline = self._build_pipeline(self._model, self._target_columns)
        return self

    def _predict(self, X):
        scores = self._predict_scores(X)
        scores_np = scores.to_numpy()
        if self.anomaly_threshold is not None:
            threshold = self.anomaly_threshold
        elif self.anomaly_percentile is not None:
            threshold = np.percentile(scores_np, self.anomaly_percentile)
        anomaly_indices = np.where(scores_np > threshold)[0]
        if len(anomaly_indices) == 0:
            return self._empty_sparse()
        return pd.Series(anomaly_indices, dtype="int64")

    def _predict_scores(self, X):
        # predict_scores does not call self._check_X(X)
        # so, calling here manually
        if isinstance(X, pd.Series):
            X = self._check_X(X)

        result = self._run_pipeline(X)
        scores = np.asarray(result["anomaly_score"]).reshape(-1)

        # Align any padded pipeline output back to X length.
        n = len(X)
        if scores.size > n:
            scores = scores[-n:]
        elif scores.size < n:
            scores = np.pad(
                scores, (n - scores.size, 0), mode="constant", constant_values=0.0
            )

        return pd.Series(scores.astype("float64", copy=False), index=X.index)

    def _transform_scores(self, X):
        scores = self._predict_scores(X)
        return pd.DataFrame(scores, columns=["scores"], index=X.index)

    def _run_pipeline(self, X):
        pipeline_df = X.copy()
        if isinstance(X.index, pd.DatetimeIndex):
            pipeline_df[TIMESTAMP_COL] = X.index
        else:
            pipeline_df[TIMESTAMP_COL] = pd.date_range(
                start="2020-01-01", periods=len(pipeline_df), freq="D"
            )

        # The tsfm_public pipeline expects at least (context_length + 1) rows
        required_len = int(getattr(self._model.config, "context_length", 0)) + 1
        if required_len > 1 and len(pipeline_df) < required_len:
            pad_len = required_len - len(pipeline_df)
            first_row = pipeline_df.iloc[[0]].drop(
                columns=[TIMESTAMP_COL], errors="ignore"
            )
            pad_block = pd.concat([first_row] * pad_len, ignore_index=True)

            # Build timestamps for the padded prefix.
            if isinstance(X.index, pd.DatetimeIndex):
                freq = X.index.inferred_freq or "D"
                first_ts = pd.to_datetime(
                    np.asarray(pipeline_df[TIMESTAMP_COL]).reshape(-1)[0]
                )
                start = first_ts - pd.tseries.frequencies.to_offset(freq) * pad_len
                pad_ts = pd.date_range(start=start, periods=pad_len, freq=freq)
            else:
                first_ts = pd.to_datetime(
                    np.asarray(pipeline_df[TIMESTAMP_COL]).reshape(-1)[0]
                )
                start = first_ts - pd.Timedelta(days=pad_len)
                pad_ts = pd.date_range(start=start, periods=pad_len, freq="D")

            pad_block[TIMESTAMP_COL] = pad_ts
            pipeline_df = pd.concat([pad_block, pipeline_df], axis=0)

        result = self._pipeline(
            pipeline_df,
            batch_size=self.batch_size,
            predictive_score_smoothing=self.predictive_score_smoothing,
            target_columns=self._target_columns,
        )
        return result

    def _build_pipeline(self, model, target_columns):
        from tsfm_public.toolkit.time_series_anomaly_detection_pipeline import (
            TimeSeriesAnomalyDetectionPipeline,
        )

        return TimeSeriesAnomalyDetectionPipeline(
            model,
            timestamp_column=TIMESTAMP_COL,
            target_columns=target_columns,
            prediction_mode=self._prediction_mode,
            aggregation_length=self.aggregation_length,
            aggr_function=self.aggr_function,
            smoothing_length=self.smoothing_length,
            least_significant_scale=self.least_significant_scale,
            least_significant_score=self.least_significant_score,
        )

    def _get_unique_model_key(self):
        key_items = {
            "model_path": self.model_path,
            "revision": self.revision,
            "num_input_channels": self._num_input_channels,
            "mask_type": self.mask_type,
            "device": self._device,
            **self._config,
        }
        return str(sorted(key_items.items()))

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return ``"default"`` set.

        Returns
        -------
        params : dict or list of dict
        """
        params1 = {
            "batch_size": 32,
            "aggregation_length": 32,
            "anomaly_threshold": 0.9,
        }
        params2 = {
            **params1,
            "anomaly_threshold": None,
            "anomaly_percentile": 95.0,
            "prediction_mode": ["time"],
        }
        return [params1, params2]


@_multiton
class _CachedTSPulseModel:
    """Cached TSPulse model shared across detector instances with the same key."""

    def __init__(self, key: str, detector: "TSPulseAnomalyDetector"):
        self.key = key
        self.detector = detector
        self.model = None

    def load_from_checkpoint(self):
        if self.model is not None:
            return self.model

        from tsfm_public.models.tspulse.modeling_tspulse import TSPulseForReconstruction

        d = self.detector
        model_config = _DEFAULT_MODEL_CONFIG.copy()
        model_config.update(d._config)
        model_config["num_input_channels"] = d._num_input_channels
        kwargs = {
            "revision": d.revision,
            "mask_type": d.mask_type,
            **model_config,
        }
        self.model = TSPulseForReconstruction.from_pretrained(d.model_path, **kwargs)
        return self.model.to(d._device)
