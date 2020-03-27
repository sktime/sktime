import numpy as np
import pandas as pd

from sktime.container import TimeDtype, TimeArray, TimeSeries
from sktime.container.utils import convert_to_timearray

# Pandas imports
from typing import (
    Optional,
    Type
)


# Main TimeFrame class--------------------------------------------------------------------------------------------------

class TimeFrame(pd.DataFrame):
    """
    A TimeFrame object is a pandas.DataFrame that has one or more columns
    containing time series.
    """

    @property
    def _constructor(self) -> Type["TimeFrame"]:
        return TimeFrame

    _constructor_sliced: Type[TimeSeries] = TimeSeries

    @property
    def _constructor_expanddim(self):
        raise NotImplementedError("Not supported for TimeFrames!")

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self,
                 data=None,
                 index= None,
                 columns=None,
                 copy=False):

        if isinstance(data, dict):
            for key in data.keys():
                data[key] = convert_to_timearray(data[key])

        elif isinstance(data, TimeArray):
            if columns is None:
                data = {0: data}
            elif not pd._lib.is_scalar(columns) and len(columns) == 1:
                data = {columns[0]: data}
            elif pd._lib.is_scalar(columns):
                raise TypeError(f"'columns' must be a collection of some kind, {repr(columns)} was passed")
            else:
                raise ValueError(f"Only a column index of length 1 allowed if 'data' is provided as TimeArray, " 
                                 f"got {columns}")

        elif isinstance(data, (pd.DataFrame, pd.Series)):
            if copy:
                data = data.copy(True)
                copy = False  # avoid a second copy in the pandas.DataFrame constructor

            if isinstance(data, pd.Series):
                data = pd.DataFrame(data, columns)

            for col in data.columns:
                data[col] = convert_to_timearray(data[col])  # TODO: ExtensionArray data gets copied again by DataFrame contructor,
                                                              #       see whether that can be avoided
            data = data._data

        elif isinstance(data, np.ndarray) and data.ndim == 3:
            if columns is None:
                columns = [i for i in range(data.shape[0])]
            data = {key: convert_to_timearray(val) for key, val in zip(columns, data)}

        super(TimeFrame, self).__init__(data, index, columns, copy=copy)

    def __getitem__(self, key):
        result = super(TimeFrame, self).__getitem__(key)

        ts_idxs = [isinstance(d, TimeDtype) for d in self.dtypes]
        ts_cols = self.columns[ts_idxs]

        if isinstance(key, str) and key in ts_cols:
            result.__class__ = TimeSeries
        elif isinstance(result, pd.DataFrame):
            result.__class__ = TimeFrame
        return result

    def to_pandas(self, inplace=False):
        if inplace:
            self.__class__ = pd.DataFrame
            return self
        else:
            return pd.DataFrame(self)

    def tabularise(self):
        return pd.concat([i.tabularise() if isinstance(i, TimeSeries) else i for _, i in self.items()], axis=1)
