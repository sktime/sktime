# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Falcon-TST forecaster for ``sktime``."""

__author__ = ["geetu040"]

__all__ = ["FalconTSTForecaster"]

from sktime.forecasting.base import BaseForecaster


class FalconTSTForecaster(BaseForecaster):
    """Falcon-TST forecaster via Hugging Face ``transformers``."""

    _tags = {
        "y_inner_mtype": "pd.DataFrame",
        "capability:multivariate": True,
        "capability:exogenous": False,
        "requires-fh-in-fit": False,
        "capability:insample": False,
        "capability:pred_int": False,
        "capability:pretrain": False,
        "authors": ["geetu040"],
        "maintainers": ["geetu040"],
        "python_dependencies": ["torch", "transformers", "einops"],
        "tests:vm": True,
    }

    def __init__(
        self,
        model_path="ant-intl/Falcon-TST_Large",
        config=None,
        device_map="cpu",
        dtype=None,
        quantization_config=None,
        revin=True,
    ):
        self.model_path = model_path
        self.config = config
        self.device_map = device_map
        self.dtype = dtype
        self.quantization_config = quantization_config
        self.revin = revin

        super().__init__()

    def _fit(self, y, X=None, fh=None):
        """Fit forecaster to training data."""
        raise NotImplementedError

    def _predict(self, fh, X):
        """Forecast time series at future horizon."""
        raise NotImplementedError
