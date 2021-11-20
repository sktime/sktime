# -*- coding: utf-8 -*-
__author__ = ["chrisholder", "TonyBagnall"]
__all__ = ["TimeSeriesPanel"]

from typing import Union

import numpy as np
import pandas as pd

TimeSeriesPanel = Union[pd.Dataframe, np.ndarray]
