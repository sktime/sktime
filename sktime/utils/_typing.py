from typing import Union, Tuple
import pandas as pd
import numpy as np
from numpy import ndarray

__all__ = [
            "Dataset",
            "Primitive",
            "Primitives",
            "Tabular",
            "UnivariateSeries",
            "MultivariateSeries",
            "Series",
            "Panel"
 ]

Dataset = Union[Tuple[pd.DataFrame, ndarray], pd.DataFrame]

# single/multiple primitives
Primitive = Union[np.int, int, np.float, float, str]
Primitives = np.ndarray

# tabular/cross-sectional data
Tabular = Union[pd.DataFrame, np.ndarray]  # 2d arrays

# univariate/multivariate series
UnivariateSeries = Union[pd.Series, np.ndarray]
MultivariateSeries = Union[pd.DataFrame, np.ndarray]
Series = Union[UnivariateSeries, MultivariateSeries]

# panel/longitudinal/series-as-features data
Panel = Union[pd.DataFrame, np.ndarray]  # 3d or nested array
