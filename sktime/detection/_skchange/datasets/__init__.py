"""Datasets and dataset generators for skchange."""

from ._data_loaders import load_hvac_system_data
from ._generate import (
    generate_piecewise_data,
)
from ._generate_linear_trend import (
    generate_continuous_piecewise_linear_data,
    generate_continuous_piecewise_linear_signal,
)
from ._generate_normal import (
    generate_alternating_data,
    generate_anomalous_data,
    generate_changing_data,
    generate_piecewise_normal_data,
)
from ._generate_regression import generate_piecewise_regression_data

DATA_LOADERS = [
    load_hvac_system_data,
]
GENERATORS = [
    generate_continuous_piecewise_linear_data,
    generate_piecewise_data,
    generate_piecewise_normal_data,
    generate_piecewise_regression_data,
]
OLD_GENERATORS = [
    generate_alternating_data,
    generate_anomalous_data,
    generate_changing_data,
    generate_continuous_piecewise_linear_signal,
]

__all__ = [
    "DATA_LOADERS",
    "GENERATORS",
    "OLD_GENERATORS",
]
