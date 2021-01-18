# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from typing import Callable, List, Tuple, Union, Mapping

# General Purpose
Data_Frame = pd.DataFrame
Series = pd.Series
Numpy_Array = np.ndarray
Tuple_Of_Numpy = Tuple[Numpy_Array, ...]

# Cluster specific
Metric_Function = Callable[[Numpy_Array, Numpy_Array, float], Numpy_Array]
Metric_Parameter = Union[Metric_Function, str]
Metric_Function_Dict = Mapping[str, Metric_Function]
Data_Parameter = Union[Numpy_Array, Data_Frame]
Data_Parameter_Arr = List[Data_Parameter]
