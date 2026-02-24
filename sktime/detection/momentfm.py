"""Interface for the momentfm anomaly detector."""

from skbase.utils.dependencies import _check_soft_dependencies

from sktime.detection.base import BaseDetector
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
DataLoader = _safe_import("torch.utils.data.DataLoader")
Dataset = _safe_import("torch.utils.data.Dataset")
empty_cache = _safe_import("torch.cuda.empty_cache")

if _check_soft_dependencies("transformers", severity="none"):
    from sktime.libs.momentfm import MOMENTPipeline
    from sktime.libs.momentfm.common import TASKS


class MomentFMDetector(BaseDetector):
    """Anomaly detector using the MomentFM foundation model.

    MomentFM is a collection of open source foundation models for general
    purpose time series analysis. This interface uses MomentFM in
    reconstruction mode - the model tries to reconstruct the input series,
    and points where reconstruction error is high are flagged as anomalies.

    Unlike the forecasting and classification variants, this detector does
    NOT require fine-tuning. The pre-trained weights are used as-is for
    zero-shot anomaly detection.

    For more information see:
    https://github.com/moment-timeseries-foundation-model/moment

    For licensing information:
    https://huggingface.co/AutonLab/MOMENT-1-large

    Parameters
    ----------
    pretrained_model_name_or_path : str
        Path to the pretrained MomentFM model or a HuggingFace model id.
        Default is "AutonLab/MOMENT-1-large".

    anomaly_criterion : str, default="mse"
        The reconstruction error metric used to compute anomaly scores.
        Supported values are "mse" and "mae".

    threshold_percentile : float, default=90
        Percentile of anomaly scores above which a point is labelled as
        an anomaly. E.g. 90 means the top 10% of scores are anomalies.

    seq_len : int, default=512
        Input sequence length. MomentFM requires windows of exactly 512
        timesteps. Shorter series are zero-padded with an appropriate mask.

    batch_size : int, default=32
        Number of windows to process in each forward pass.

    device : str, default="auto"
        Torch device to run inference on. "auto" will pick up GPU if
        available, otherwise falls back to CPU.

    config : dict, default=None
        Optional dict of additional MomentFM model kwargs. Values here
        override individual parameter arguments when both are given.

    to_cpu_after_predict : bool, default=False
        If True, moves the model back to CPU after each predict call.
        Helps with memory when running multiple detectors.

    References
    ----------
    Paper: https://arxiv.org/abs/2402.03885
    Github: https://github.com/moment-timeseries-foundation-model/moment/tree/main

    Examples
    --------
    >>> from sktime.detection.momentfm import MomentFMDetector
    >>> import pandas as pd, numpy as np
    >>> rng = np.random.default_rng(42)
    >>> X = pd.DataFrame(rng.standard_normal(600))
    >>> X.iloc[100] = 50  # inject anomaly
    >>> detector = MomentFMDetector(threshold_percentile=95)
    >>> detector.fit(X)  # doctest: +SKIP
    >>> labels = detector.predict(X)  # doctest: +SKIP
    """

    _tags = {
        "authors": ["priyanshuharshbodhi1"],
        "maintainers": ["priyanshuharshbodhi1"],
        "task": "anomaly_detection",
        "learning_type": "unsupervised",
        "X_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:missing_values": False,
        "python_dependencies": [
            "torch",
            "huggingface-hub",
            "hf-xet",
            "transformers",
        ],
        "python_version": ">= 3.10",
        # testing configuration
        # ---------------------
        "tests:vm": True,
        "tests:libs": ["sktime.libs.momentfm"],
    }

    def __init__(
        self,
        pretrained_model_name_or_path="AutonLab/MOMENT-1-large",
        anomaly_criterion="mse",
        threshold_percentile=90,
        seq_len=512,
        batch_size=32,
        device="auto",
        config=None,
        to_cpu_after_predict=False,
    ):
        self.pretrained_model_name_or_path = pretrained_model_name_or_path
        self.anomaly_criterion = anomaly_criterion
        self.threshold_percentile = threshold_percentile
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device
        self.config = config
        self._config = config if config is not None else {}
        self.to_cpu_after_predict = to_cpu_after_predict
        super().__init__()

    def _fit(self, X, y=None):
        """Load MomentFM in reconstruction mode.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to fit on. Not used for training - model weights
            are loaded from the pretrained checkpoint. X is only needed
            so the base class fit/predict contract is satisfied.
        y : ignored

        Returns
        -------
        self
        """
        self._pretrained_model_name_or_path = self._config.get(
            "pretrained_model_name_or_path", self.pretrained_model_name_or_path
        )
        self._device = self._config.get("device", self.device)
        self._device = _check_device(self._device)

        self.model = MOMENTPipeline.from_pretrained(
            self._pretrained_model_name_or_path,
            model_kwargs={
                "task_name": TASKS.RECONSTRUCTION,
                "device": self._device,
            },
        )
        self.model.init()

        return self

    def _predict(self, X):
        raise NotImplementedError

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        raise NotImplementedError


def _check_device(device):
    """Resolve device string to a valid torch device.

    Mirrors the same helper used in MomentFMForecaster and
    MomentFMClassifier - kept here to avoid a cross-module import.
    """
    if device == "auto":
        return device
    mps = False
    cuda = False
    if device == "mps":
        from torch.backends.mps import is_available, is_built

        if not is_available():
            if not is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        else:
            _device = "mps"
            mps = True
    elif device in ("gpu", "cuda"):
        from torch.cuda import is_available

        if is_available():
            _device = "cuda"
            cuda = True
    elif "cuda" in device:
        from torch.cuda import is_available

        if is_available():
            _device = device
            cuda = True
    if mps or cuda:
        return _device
    return "cpu"
