"""Residual Boosting Forecasters."""

from sktime.forecasting.residual_booster._residual_booster import (
    ResidualBoostingForecaster,
)
from sktime.forecasting.residual_booster._residual_chain_booster import (
    ResidualBoostingChainForecaster,
)

__all__ = ["ResidualBoostingForecaster", "ResidualBoostingChainForecaster"]
