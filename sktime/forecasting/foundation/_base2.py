"""Shared base class for foundation-model forecasters."""

from sklearn.utils import check_random_state

from sktime.forecasting.base import BaseForecaster


class BaseFoundationForecaster(BaseForecaster):
    """Shared base class for pretrained/foundation forecasting models."""

    def __init__(
        self,
        model_path=None,
        config=None,
        load_kwargs=None,
        quantization_config=None,
        device=None,
        forward_kwargs=None,
        random_state=None,
    ):
        self.model_path = model_path
        self.config = config
        self.load_kwargs = load_kwargs
        self.quantization_config = quantization_config
        self.device = device
        self.forward_kwargs = forward_kwargs
        self.random_state = random_state

        super().__init__()

    def __post_init__(self):
        """Post-init constructor logic for shared foundation-model parameters."""
        self.random_state_ = check_random_state(self.random_state)
