# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""pytorch-forecasting v2 Temporal Fusion Transformer (TFT) estimator."""

import functools
from typing import Any

from sktime.forecasting.base.adapters._pytorchforecasting_v2 import (
    _PytorchForecastingAdapterV2,
)

__author__ = ["vedantag17"]


class PytorchForecastingTFTV2(_PytorchForecastingAdapterV2):
    """pytorch-forecasting v2 Temporal Fusion Transformer (TFT) model.

    The Temporal Fusion Transformer combines high-performance multi-horizon
    forecasting with interpretable insights into temporal dynamics. It uses
    specialized components including variable selection networks, static
    covariate encoders, gating mechanisms, and multi-head attention to
    capture complex temporal patterns.


    Parameters
    ----------
    model_params : dict[str, Any] or None, default=None
        Parameters for the TFT model constructor.
        All model-specific hyper-parameters are passed through this dict
        to the underlying ``pytorch_forecasting`` TFT class.

    data_module_params : dict[str, Any] or None, default=None
        Parameters for ``EncoderDecoderTimeSeriesDataModule``.

    trainer_params : dict[str, Any] or None, default=None
        Parameters for ``lightning.pytorch.Trainer``.

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
        from pytorch_forecasting.models.temporal_fusion_transformer._tft_v2 import (
            TFT,
        )

        return TFT

    @functools.cached_property
    def algorithm_parameters(self) -> dict:
        """Get keyword parameters for the TFT class.

        Returns
        -------
        dict
            keyword arguments for the underlying algorithm class
        """
        return {}

    @functools.cached_property
    def data_module_class(self):
        """Return the DataModule class for TFT."""
        from pytorch_forecasting.data.data_module._encoder_decoder_data_module import (
            EncoderDecoderTimeSeriesDataModule,
        )

        return EncoderDecoderTimeSeriesDataModule

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
