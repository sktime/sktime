# -*- coding: utf-8 -*-
"""Base types for clustering"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = [
    "DataFrame",
    "Series",
    "NumpyArray",
    "TupleOfNumpy",
    "NumpyRandomState",
    "NumpyOrDF",
    "MetricFunction",
    "MetricParameter",
    "MetricFunctionDict",
    "DataParameter",
    "DataParameterArr",
    "InitAlgo",
    "InitAlgoDict",
    "AveragingAlgo",
    "AveragingAlgoDict",
    "CenterCalculatorFunc",
]

import pandas as pd
from typing import Callable, List, Tuple, Union, Mapping
from sktime.clustering.base.base import (
    NumpyArray,
    DataFrame,
    NumpyOrDF,
    BaseClusterCenterInitializer,
    BaseClusterAverage,
    NumpyRandomState,
    CenterCalculatorFunc,
)

# General Purpose
Series = pd.Series
TupleOfNumpy = Tuple[NumpyArray, ...]

# Cluster specific
MetricFunction = Callable[[NumpyArray, NumpyArray, float], NumpyArray]
MetricParameter = Union[MetricFunction, str]
MetricFunctionDict = Mapping[str, MetricFunction]

DataParameter = Union[NumpyArray, DataFrame]
DataParameterArr = List[DataParameter]

InitAlgo = Union[str, BaseClusterCenterInitializer]
InitAlgoDict = Mapping[str, BaseClusterCenterInitializer]

AveragingAlgo = Union[str, BaseClusterAverage]
AveragingAlgoDict = Mapping[str, AveragingAlgo]
