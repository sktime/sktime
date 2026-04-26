# Use future annotations for better handling of forward references.
from __future__ import annotations

import sys
from enum import Enum
from numbers import Integral
from typing import Any, List, Literal, Optional, Union

from numpy.random import Generator
from packaging.specifiers import SpecifierSet

# Define model and block compressor types using Literal for clearer enum-style typing.
ModelTypesWithoutArch = Literal["ar", "arima", "sarima", "var"]

ModelTypes = Literal["ar", "arima", "sarima", "var", "arch"]

BlockCompressorTypes = Literal[
    "first",
    "middle",
    "last",
    "mean",
    "mode",
    "median",
    "kmeans",
    "kmedians",
    "kmedoids",
]


class DistributionTypes(Enum):
    """
    Enumeration of supported distribution types for block length sampling.
    """

    NONE = "none"
    POISSON = "poisson"
    EXPONENTIAL = "exponential"
    NORMAL = "normal"
    GAMMA = "gamma"
    BETA = "beta"
    LOGNORMAL = "lognormal"
    WEIBULL = "weibull"
    PARETO = "pareto"
    GEOMETRIC = "geometric"
    UNIFORM = "uniform"


# Check Python version for compatibility issues.
sys_version = sys.version.split(" ")[0]
new_typing_available = sys_version in SpecifierSet(">=3.10")


def FittedModelTypes() -> tuple:
    """
    Return a tuple of fitted model types for use in isinstance checks.

    Returns
    -------
        tuple: A tuple containing the result wrapper types for fitted models.
    """
    from arch.univariate.base import ARCHModelResult
    from statsmodels.tsa.ar_model import AutoRegResultsWrapper
    from statsmodels.tsa.arima.model import ARIMAResultsWrapper
    from statsmodels.tsa.statespace.sarimax import SARIMAXResultsWrapper
    from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper

    fmt = (
        AutoRegResultsWrapper,
        ARIMAResultsWrapper,
        SARIMAXResultsWrapper,
        VARResultsWrapper,
        ARCHModelResult,
    )
    return fmt


# Define complex type conditions using the Python 3.10 union operator if available.
if new_typing_available:
    OrderTypesWithoutNone = Union[
        Integral,
        List[Integral],
        tuple[Integral, Integral, Integral],
        tuple[Integral, Integral, Integral, Integral],
    ]
    OrderTypes = Optional[OrderTypesWithoutNone]

    RngTypes = Optional[Union[Generator, Integral]]

else:
    OrderTypesWithoutNone = Any
    OrderTypes = Any
    RngTypes = Any
