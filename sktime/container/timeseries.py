import numpy as np
import pandas as pd
from typing import Type

from sktime.container.array import TimeDtype
from sktime.container.utils import convert_to_timearray


class TimeSeries(pd.Series):
    """
    A Series object designed to store time series objects.

    Parameters
    ----------
    data : array-like, dict, scalar value
        The geometries to store in the GeoSeries.
    index : array-like or Index
        The row index for the TimeSeries.
    time_index : array-like or Index (optional)
        The index denoting the relative position of each observation in each time series
    kwargs
        Additional arguments passed to the Series constructor,
         e.g. ``name``.

    See Also
    --------
    TimeSeriesFrame
    pandas.Series

    """

    _metadata = ["name"]

    @property
    def _constructor(self) -> Type["TimeSeries"]:
        return TimeSeries

    @property
    def _constructor_expanddim(self) -> Type["TimeFrame"]:
        from sktime.container.timeframe import TimeFrame

        return TimeFrame

    def __init__(self, data=None, index=None, time_index=None, **kwargs):
        name = kwargs.pop("name", None)

        if isinstance(data, (pd.DataFrame, pd.Series, np.ndarray)):
            data = convert_to_timearray(data, time_index)

        super(TimeSeries, self).__init__(data, index=index, name=name, **kwargs)

    # -------------------------------------------------------------------------
    # time series functionality
    # -------------------------------------------------------------------------


    def is_timedata(self):
        return isinstance(self.dtype, TimeDtype)

    def tabularise(self, return_array=False):
        if self.is_timedata():
            return self._values.tabularise(self.name, return_array)
        else:
            if return_array:
                return self.to_numpy()
            else:
                return self

    def tabularize(self, return_array=False):
        return self.tabularise(return_array)

    def slice_time(self, time_index, inplace=False):
        if inplace:
            # TODO: enable inplace slicing
            raise NotImplementedError("inplace slicing of time series is not supported yet")

        return TimeSeries(self._values.slice_time(time_index), index=self.index, name=self.name)

    @property
    def has_common_index(self):
        return self._values.has_common_index

    @property
    def time_index(self):
        return self._values.time_index