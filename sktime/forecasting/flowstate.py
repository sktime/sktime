# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Implements FlowState for forecasting."""

__author__ = ["Faakhir30"]
__all__ = ["FlowStateForecaster"]

import numpy as np

from sktime.forecasting.base import _GlobalForecastingDeprecationMixin
from sktime.forecasting.foundation import (
    BaseFoundationForecaster,
    ForecastResult,
    FoundationModelSpec,
    ModelHandle,
)
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")


class FlowStateForecaster(_GlobalForecastingDeprecationMixin, BaseFoundationForecaster):
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
        "capability:exogenous": False,
        "capability:multivariate": False,
        "capability:unequal_length": True,
        "capability:insample": False,
        "capability:pred_int": True,
        "capability:pred_int:insample": False,
        "capability:global_forecasting": True,
        "requires-fh-in-fit": False,
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

        model_spec = FoundationModelSpec(
            model_path=model_path,
            revision=revision,
            config=config,
            device="auto",
            predict_extra_kwargs={
                "scale_factor": scale_factor,
                "batch_first": batch_first,
                "prediction_type": prediction_type,
            },
        )
        super().__init__(model_spec=model_spec)

    def _load_model(self):
        """Load a FlowState checkpoint into a cacheable model handle."""
        from tsfm_public import FlowStateForPrediction

        model_spec = self.model_spec
        model = FlowStateForPrediction.from_pretrained(
            model_spec.model_path,
            revision=model_spec.revision,
            **(model_spec.config or {}),
            **model_spec.load_extra_kwargs,
        )
        model = model.to(model_spec.device)

        return ModelHandle(model=model)

    def _inference(
        self,
        handle,
        context_y,
        context_X,
        future_X,
        pred_len,
        fh,
        alpha=None,
    ):
        """Run the FlowState forward pass and normalize its outputs."""
        model = handle.model
        predict_kwargs = self.model_spec.predict_extra_kwargs
        past = torch.tensor(
            context_y.iloc[:, 0].to_numpy(dtype=np.float32).reshape(1, -1, 1),
            dtype=model.dtype,
            device=model.device,
        )
        output = model(
            past_values=past,
            prediction_length=pred_len,
            **predict_kwargs,
        )

        values = output.prediction_outputs.detach().cpu().numpy()[0]
        if values.ndim == 1:
            values = values[:, np.newaxis]

        quantiles = None
        if alpha is not None:
            native_quantiles = (
                output.quantile_outputs.detach().cpu().numpy()[0, :, :, 0]
            )
            model_quantiles = np.asarray(model.config.quantiles, dtype=float)
            quantiles = {
                quantile: np.asarray(
                    [
                        np.interp(
                            quantile,
                            model_quantiles,
                            native_quantiles[:, timepoint],
                        )
                        for timepoint in range(pred_len)
                    ]
                ).reshape(-1, 1)
                for quantile in alpha
            }

        point_key = (
            "median" if predict_kwargs["prediction_type"] == "median" else "mean"
        )
        return ForecastResult(**{point_key: values}, quantiles=quantiles)

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
