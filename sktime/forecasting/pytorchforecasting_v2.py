# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Interfaces to pytorch-forecasting v2 estimators.

This module contains sktime estimators that wrap PTF v2 models.
"""

import functools
from typing import Any

from sktime.forecasting.base.adapters._pytorchforecasting_v2 import (
    _PytorchForecastingAdapterV2,
)

__author__ = ["vedantag17"]


TslibBatchTFT = None


def _tslib_batch_to_tft_batch(x: dict[str, Any]) -> dict[str, Any]:
    """Convert generic TslibDataModule batch keys to TFT v2 batch keys."""
    if "encoder_cont" in x:
        return x

    encoder_cont_parts = []
    history_cont = x.get("history_cont")
    history_target = x.get("history_target")
    if history_cont is not None and history_cont.size(-1) > 0:
        encoder_cont_parts.append(history_cont)
    if history_target is not None and history_target.size(-1) > 0:
        encoder_cont_parts.append(history_target)

    if encoder_cont_parts:
        import torch

        encoder_cont = torch.cat(encoder_cont_parts, dim=-1)
    elif history_cont is not None:
        encoder_cont = history_cont
    else:
        raise KeyError(
            "TFT input requires either 'encoder_cont' or Tslib "
            "'history_cont'/'history_target' keys."
        )

    tft_x = {
        "encoder_cont": encoder_cont,
        "encoder_cat": x["history_cat"],
        "decoder_cont": x["future_cont"],
        "decoder_cat": x["future_cat"],
    }

    for key in (
        "static_categorical_features",
        "static_continuous_features",
        "target_scale",
    ):
        if key in x:
            tft_x[key] = x[key]

    return tft_x


def _get_tslib_batch_tft_class():
    """Return a pickleable TFT subclass accepting TslibDataModule batches."""
    global TslibBatchTFT

    if TslibBatchTFT is None:
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import (
            TFT,
        )

        class _TslibBatchTFT(TFT):
            """TFT variant accepting TslibDataModule's generic batch schema."""

            def forward(self, x: dict[str, Any]) -> dict[str, Any]:
                """Run TFT on either native TFT or generic Tslib batch keys."""
                return super().forward(_tslib_batch_to_tft_batch(x))

        _TslibBatchTFT.__name__ = "TslibBatchTFT"
        _TslibBatchTFT.__qualname__ = "TslibBatchTFT"
        _TslibBatchTFT.__module__ = __name__
        TslibBatchTFT = _TslibBatchTFT

    return TslibBatchTFT


class PytorchForecastingTFTV2(_PytorchForecastingAdapterV2):
    """pytorch-forecasting v2 Temporal Fusion Transformer (TFT) model.

    The Temporal Fusion Transformer combines high-performance multi-horizon
    forecasting with interpretable insights into temporal dynamics. It uses
    specialized components including variable selection networks, static
    covariate encoders, gating mechanisms, and multi-head attention to
    capture complex temporal patterns.

    This estimator uses the PTF v2 data pipeline (``TimeSeries`` D1 +
    ``TslibDataModule`` D2) and ``TslibBaseModel`` model hierarchy.

    Parameters
    ----------
    model_params : dict[str, Any] or None, default=None
        Parameters for the TFT model constructor.
        All model-specific hyper-parameters are passed through this dict
        to the underlying ``pytorch_forecasting`` TFT class. The adapter
        automatically handles ``loss``, ``metadata``, ``optimizer``, and
        ``lr_scheduler`` parameters.

    data_module_params : dict[str, Any] or None, default=None
        Parameters for ``TslibDataModule``.
        ``context_length`` and ``prediction_length`` will be auto-inferred
        from the data and ``fh`` if not provided.
        ``time_series_dataset`` is constructed automatically from sktime data.

        **NOTE**: This replaces the v1 three-way split of ``dataset_params``,
        ``train_to_dataloader_params``, and ``validation_to_dataloader_params``
        because v2's ``TslibDataModule`` manages both datasets and dataloaders
        internally.

    trainer_params : dict[str, Any] or None, default=None
        Parameters for ``lightning.pytorch.Trainer``.
        Example: ``{"max_epochs": 10, "accelerator": "cpu"}``.

    broadcasting : bool, default=False
        If True, fall back to per-series fitting instead of global.

    Examples
    --------
    >>> from sktime.forecasting.base import ForecastingHorizon
    >>> from sktime.forecasting.pytorchforecasting_v2 import (
    ...     PytorchForecastingTFTV2,
    ... )
    >>> from sktime.utils._testing.hierarchical import _make_hierarchical
    >>> # generate random data
    >>> data = _make_hierarchical(
    ...     hierarchy_levels=(2, 10), max_timepoints=30, min_timepoints=30,
    ...     n_columns=2,
    ... )
    >>> y = data["c1"].to_frame()
    >>> fh = ForecastingHorizon(range(1, 4), is_relative=True)
    >>> model = PytorchForecastingTFTV2(
    ...     trainer_params={"max_epochs": 1, "limit_train_batches": 2,
    ...                     "enable_checkpointing": False, "logger": False},
    ... )
    >>> model.fit(y=y, fh=fh)  # doctest: +SKIP
    >>> y_pred = model.predict(fh)  # doctest: +SKIP

    References
    ----------
    .. [1] https://arxiv.org/abs/1912.09363
    .. [2] https://pytorch-forecasting.readthedocs.io/en/latest/api/
       pytorch_forecasting.models.temporal_fusion_transformer._tft_v2.TFT.html
    """

    _tags = {
        "capability:global_forecasting": True,
        "capability:insample": False,
        "X-y-must-have-same-index": True,
        "capability:multivariate": False,
        "capability:pred_int": False,
        "capability:unequal_length": False,
        # CI and test flags
        "tests:core": True,
        "tests:skip_all": False,
        "tests:skip_by_name": [
            "test_save_estimators_to_file",
            "test_persistence_via_pickle",
            "test_hierarchical_with_exogenous",
        ],
    }

    def __init__(
        self,
        model_params: dict[str, Any] | None = None,
        data_module_params: dict[str, Any] | None = None,
        trainer_params: dict[str, Any] | None = None,
        broadcasting: bool = False,
    ) -> None:
        super().__init__(
            model_params=model_params,
            data_module_params=data_module_params,
            trainer_params=trainer_params,
            broadcasting=broadcasting,
        )

    @functools.cached_property
    def algorithm_class(self):
        """Import underlying pytorch-forecasting v2 TFT class."""
        return _get_tslib_batch_tft_class()

    @functools.cached_property
    def algorithm_parameters(self) -> dict:
        """Get keyword parameters for the TFT class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        return {}

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return.

        Returns
        -------
        params : dict or list of dict
            Parameters to create testing instances of the class.
        """
        params = [
            {
                "trainer_params": {
                    "max_epochs": 1,
                    "limit_train_batches": 2,
                    "enable_checkpointing": False,
                    "logger": False,
                },
                "data_module_params": {
                    "context_length": 3,
                    "batch_size": 2,
                },
            },
            {
                "trainer_params": {
                    "max_epochs": 1,
                    "limit_train_batches": 2,
                    "enable_checkpointing": False,
                    "logger": False,
                },
                "data_module_params": {
                    "context_length": 5,
                    "batch_size": 2,
                },
            },
        ]

        params_broadcasting = [dict(p, **{"broadcasting": True}) for p in params]
        params_no_broadcasting = [dict(p, **{"broadcasting": False}) for p in params]
        return params_broadcasting + params_no_broadcasting
