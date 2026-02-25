"""Interface for the momentfm anomaly detector."""

import numpy as np
import pandas as pd
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
        """Load MomentFM in reconstruction mode."""
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
        """Detect anomalies by thresholding per-timestep reconstruction error.

        Parameters
        ----------
        X : pd.DataFrame
            Time series to detect anomalies in.

        Returns
        -------
        y : pd.Series with RangeIndex, dtype int64
            Sparse format: values are iloc indices of detected anomalies.
        """
        import torch

        X_vals = X.values.astype(np.float32)
        n_timepoints = len(X)

        dataset = MomentFMDetectorPytorchDataset(X_vals, seq_len=self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        score_sum = np.zeros(n_timepoints, dtype=np.float64)
        count = np.zeros(n_timepoints, dtype=np.int32)

        self.model.eval()
        self.model.to(self._device)

        with torch.no_grad():
            for batch in loader:
                x_enc = batch["x_enc"].to(self._device)
                input_mask = batch["input_mask"].to(self._device)
                start_indices = batch["start_idx"].numpy()

                outputs = self.model.detect_anomalies(
                    x_enc=x_enc,
                    input_mask=input_mask,
                    anomaly_criterion=self.anomaly_criterion,
                )
                scores = outputs.anomaly_scores.cpu().numpy()
                scores = scores.mean(axis=1)

                for i, start in enumerate(start_indices):
                    actual_len = int(input_mask[i].sum().item())
                    end = start + actual_len
                    score_sum[start:end] += scores[i, :actual_len]
                    count[start:end] += 1

        count = np.where(count == 0, 1, count)
        anomaly_scores = score_sum / count

        threshold = np.percentile(anomaly_scores, self.threshold_percentile)
        anomaly_ilocs = np.where(anomaly_scores >= threshold)[0]

        if self.to_cpu_after_predict:
            self.model.to("cpu")
            empty_cache()

        return pd.Series(anomaly_ilocs, dtype="int64")

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator."""
        params1 = {
            "to_cpu_after_predict": True,
            "threshold_percentile": 90,
            "batch_size": 8,
        }
        params2 = {
            "anomaly_criterion": "mae",
            "to_cpu_after_predict": True,
            "threshold_percentile": 95,
            "batch_size": 16,
        }
        return [params1, params2]


class MomentFMDetectorPytorchDataset(Dataset):
    """Windowed pytorch dataset for MomentFM anomaly detection.

    Slides a window of size ``seq_len`` over the input time series with
    stride 1, returning x_enc, input_mask and start_idx per window.

    Parameters
    ----------
    X : np.ndarray of shape (n_timepoints, n_channels)
        The time series values.
    seq_len : int
        Window length.
    """

    _moment_seq_len = 512

    def __init__(self, X, seq_len=512):
        import torch

        self.seq_len = min(seq_len, self._moment_seq_len)
        self.n_timepoints, self.n_channels = X.shape
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        """Return number of windows."""
        if self.n_timepoints <= self.seq_len:
            return 1
        return self.n_timepoints - self.seq_len + 1

    def __getitem__(self, idx):
        """Return one window as a dict with keys x_enc, input_mask, start."""
        import torch

        end = idx + self.seq_len

        if end <= self.n_timepoints:
            window = self.X[idx:end, :]
            input_mask = torch.ones(self._moment_seq_len)
            if self.seq_len < self._moment_seq_len:
                pad_len = self._moment_seq_len - self.seq_len
                padding = torch.zeros(pad_len, self.n_channels)
                window = torch.cat([window, padding], dim=0)
                input_mask[self.seq_len :] = 0.0
        else:
            window = self.X[idx:, :]
            actual_len = window.shape[0]
            pad_len = self._moment_seq_len - actual_len
            padding = torch.zeros(pad_len, self.n_channels)
            window = torch.cat([window, padding], dim=0)
            input_mask = torch.zeros(self._moment_seq_len)
            input_mask[:actual_len] = 1.0

        x_enc = window.T

        return {
            "x_enc": x_enc,
            "input_mask": input_mask,
            "start_idx": idx,
        }


def _check_device(device):
    """Resolve device string to a valid torch device."""
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
