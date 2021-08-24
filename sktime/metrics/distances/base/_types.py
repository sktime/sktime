# -*- coding: utf-8 -*-
__all__ = ["SktimeSeries", "SktimeMatrix"]

from typing import Union, List
import numpy as np
import pandas as pd

SktimeSeries = Union[np.ndarray, pd.DataFrame, pd.Series, List]
SktimeMatrix = Union[np.ndarray, pd.DataFrame, List]
