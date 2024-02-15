"""Module containing adapters to other forecasting framework packages."""
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

__all__ = [
    "HCrystalBallAdapter",
    "AutoTSAdapter",
]

from sktime.forecasting.adapters._autots import AutoTSAdapter
from sktime.forecasting.adapters._hcrystalball import HCrystalBallAdapter
