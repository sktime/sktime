# -*- coding: utf-8 -*-
"""Base types for clustering"""

__author__ = ["Christopher Holder", "Tony Bagnall"]
__all__ = [
    "Data_Frame",
    "Series",
    "Numpy_Array",
    "Tuple_Of_Numpy",
    "Numpy_Or_DF",
    "Metric_Function",
    "Metric_Parameter",
    "Metric_Function_Dict",
    "Data_Parameter",
    "Data_Parameter_Arr",
]

import pandas as pd
import numpy as np
from typing import Callable, List, Tuple, Union, Mapping

# General Purpose
Data_Frame = pd.DataFrame
Series = pd.Series
Numpy_Array = np.ndarray
Tuple_Of_Numpy = Tuple[Numpy_Array, ...]
Numpy_Or_DF = Union[Data_Frame, Numpy_Array]

# Cluster specific
Metric_Function = Callable[[Numpy_Array, Numpy_Array, float], Numpy_Array]
Metric_Parameter = Union[Metric_Function, str]
Metric_Function_Dict = Mapping[str, Metric_Function]

Data_Parameter = Union[Numpy_Array, Data_Frame]
Data_Parameter_Arr = List[Data_Parameter]
