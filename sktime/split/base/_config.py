#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)

from typing import Iterator, Tuple, Union

import numpy as np
import pandas as pd

from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.base._fh import VALID_FORECASTING_HORIZON_TYPES

DEFAULT_STEP_LENGTH = 1
DEFAULT_WINDOW_LENGTH = 10
DEFAULT_FH = 1
ACCEPTED_Y_TYPES = Union[pd.Series, pd.DataFrame, np.ndarray, pd.Index]
FORECASTING_HORIZON_TYPES = Union[
    Union[VALID_FORECASTING_HORIZON_TYPES], ForecastingHorizon
]
SPLIT_TYPE = Union[
    Tuple[pd.Series, pd.Series], Tuple[pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]
]
SPLIT_ARRAY_TYPE = Tuple[np.ndarray, np.ndarray]
SPLIT_GENERATOR_TYPE = Iterator[SPLIT_ARRAY_TYPE]
PANDAS_MTYPES = ["pd.DataFrame", "pd.Series", "pd-multiindex", "pd_multiindex_hier"]
