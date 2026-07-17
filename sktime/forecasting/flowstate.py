# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements FlowState for forecasting."""

__author__ = ["Faakhir30"]
__all__ = ["FlowStateForecaster"]

import numpy as np
import pandas as pd
from skbase.utils.dependencies import _check_soft_dependencies

from sktime.forecasting.base import (
    BaseForecaster,
    ForecastingHorizon,
    _GlobalForecastingDeprecationMixin,
)
from sktime.utils.dependencies import _safe_import
from sktime.utils.singleton import _multiton

torch = _safe_import("torch")


class FlowStateForecaster(_GlobalForecastingDeprecationMixin, BaseForecaster):
    """Zero-shot forecaster wrapping IBM FlowState via granite-tsfm.

    FlowState, developed by IBM Research, is an encoder-decoder architecture,
    employing an S5-based encoder and a functional basis decoder.

    Univariate only. Implementation adapted from [1]_.

    Parameters
    ----------
    model_path : str, default="ibm-research/flowstate"
        Hugging Face model id or local path.
    revision : str, default="r1.1"
        Model revision on the Hugging Face Hub. Always forwarded to
        ``from_pretrained``; do not duplicate in ``config``.
    scale_factor : float, default=1.0
        Temporal scaling passed to the model at predict time.
    config : dict, optional, default=None
        Extra kwargs for ``FlowStateForPrediction.from_pretrained``.
    batch_first : bool, default=True
        ``past_values`` layout for the model.
    prediction_type : {"mean", "median"}, default="mean"
        Point forecast type passed to the model.

    References
    ----------
    .. [1] https://github.com/ibm-granite/granite-tsfm/blob/main/notebooks/hfdemo/flowstate_getting_started_pipeline.ipynb
    .. [2] Graf et al., FlowState: Sampling Rate Invariant Time Series Forecasting,
           arXiv:2508.05287

    Examples
    --------
    >>> from sktime.datasets import load_airline
    >>> from sktime.forecasting.flowstate import FlowStateForecaster
    >>> from sktime.split import temporal_train_test_split
    >>> y = load_airline()
    >>> y_train, _ = temporal_train_test_split(y)
    >>> f = FlowStateForecaster()  # doctest: +SKIP
    >>> f.fit(y_train)  # doctest: +SKIP
    >>> y_pred = f.predict(fh=[1, 2, 3])  # doctest: +SKIP
    """

    _tags = {
        "authors": [
            "largraf",
            "bohnstingl",
            "Angeliki Pantazi",
            "Stanisław Woźniak",
            "Faakhir30",
        ],
        "maintainers": ["Faakhir30"],
        "python_version": ">=3.11",
        "python_dependencies": [
            "granite-tsfm>=0.3.5",
            "torch",
            "transformers",
            "accelerate",
        ],
        "X_inner_mtype": "pd.DataFrame",
        "y_inner_mtype": "pd.DataFrame",
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:unequal_length": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        "requires-fh-in-fit": False,
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path: str = "ibm-research/flowstate",
        revision: str = "r1.1",
        scale_factor: float = 1.0,
        config: dict | None = None,
        batch_first: bool = True,
        prediction_type: str = "mean",
    ):
        self.model_path = model_path
        self.revision = revision
        self.scale_factor = scale_factor
        self.config = config
        self.batch_first = batch_first
        self.prediction_type = prediction_type
        self.model = None
        super().__init__()

    def __getstate__(self):
        """Get state for pickling."""
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        """Set state for unpickling."""
        self.__dict__.update(state)

    def __post_init__(self):
        """Post-initialization setup."""
        self._config = {} if self.config is None else self.config.copy()
        self._device = _resolve_device()

    def _get_unique_model_key(self):
        key_items = {
            "model_path": self.model_path,
            "revision": self.revision,
            "batch_first": self.batch_first,
            "prediction_type": self.prediction_type,
            "device": self._device,
            **self._config,
        }
        return str(sorted(key_items.items()))

    def _load_model(self):
        return _CachedFlowState(
            key=self._get_unique_model_key(),
            forecaster=self,
        ).load_from_checkpoint()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data.

        Loads the pretrained FlowState checkpoint and stores ``y`` as context
        for zero-shot prediction.

        Parameters
        ----------
        y : pd.DataFrame
            Endogenous time series (univariate, one column).
        X : pd.DataFrame, optional (default=None)
            Exogenous variables. Ignored.
        fh : ForecastingHorizon, optional (default=None)
            Forecasting horizon.

        Returns
        -------
        self
        """
        self._scale_factor = float(self.scale_factor)
        self.model = self._load_model()
        self.model.eval()
        self._context = y
        return self

    def _run(self, pred_len):
        if self.model is None:
            self.model = self._load_model()
        self.model.eval()
        past = torch.tensor(
            self._context.iloc[:, 0].to_numpy(dtype=np.float32).reshape(1, -1, 1),
            dtype=self.model.dtype,
            device=self.model.device,
        )
        with torch.inference_mode():
            return self.model(
                past_values=past,
                prediction_length=pred_len,
                batch_first=self.batch_first,
                scale_factor=self._scale_factor,
                prediction_type=self.prediction_type,
            )

    def _predict(self, fh, X=None):
        if fh is None:
            fh = self.fh
        fh_rel = fh.to_relative(self.cutoff)
        pred_len = int(np.max(fh_rel.to_numpy()))
        out = self._run(pred_len)
        values = out.prediction_outputs.detach().cpu().numpy()[0]
        if values.ndim == 1:
            values = values[:, np.newaxis]

        pred_len_out = values.shape[0]
        index = (
            ForecastingHorizon(range(1, pred_len_out + 1))
            .to_absolute(self._cutoff)
            ._values
        )
        pred_df = pd.DataFrame(values, index=index, columns=self._context.columns)
        pred_df.index.names = self._context.index.names
        pred_out = fh_rel.get_expected_pred_idx(self._context, cutoff=self.cutoff)
        return pred_df.loc[pred_df.index.isin(pred_out)]

    def _predict_quantiles(self, fh, X, alpha):
        fh_rel = fh.to_relative(self.cutoff)
        pred_len = int(np.max(fh_rel.to_numpy()))
        out = self._run(pred_len)
        q = out.quantile_outputs.detach().cpu().numpy()[0, :, :, 0]
        model_q = np.asarray(self.model.config.quantiles, dtype=float)
        rel_idx = (fh_rel.to_numpy() - 1).astype(int)

        var_name = self._context.columns[0]
        pred_index = fh.to_absolute(self.cutoff)._values
        cols = pd.MultiIndex.from_product([[var_name], alpha])
        pred_q = pd.DataFrame(index=pred_index, columns=cols)
        for a in alpha:
            pred_q[(var_name, a)] = np.array(
                [np.interp(a, model_q, q[:, i]) for i in rel_idx]
            )
        pred_q.index.names = self._context.index.names
        return pred_q

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
        return [
            {},
            {"scale_factor": 0.5},
        ]


@_multiton
class _CachedFlowState:
    """Cached FlowState model; shared across forecaster instances with the same key."""

    def __init__(self, key: str, forecaster: "FlowStateForecaster"):
        self.key = key
        self.forecaster = forecaster
        self.model = None

    def load_from_checkpoint(self):
        if self.model is not None:
            return self.model

        from tsfm_public import FlowStateForPrediction

        f = self.forecaster
        kwargs = {"revision": f.revision, "batch_first": f.batch_first}
        kwargs.update(f._config)
        self.model = FlowStateForPrediction.from_pretrained(f.model_path, **kwargs)
        return self.model.to(f._device)


def _resolve_device():
    if _check_soft_dependencies("torch", severity="none"):
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    return "cpu"
