import numpy as np
import pandas as pd

# TODO: add typing
from typing import Type

from sktime.container.array import TimeDtype
from sktime.container.utils import convert_to_timearray


class TimeSeries(pd.Series):
    """
    A Series object designed to store time series objects.

    Parameters
    ----------
    data : array-like, dict, scalar value
        The timeseries to store in the TimeSeries.
    index : array-like or Index
        The row index for the TimeSeries.
    time_index : array-like or Index (optional)
        The index denoting the relative position of each observation in each
        time series
    kwargs
        Additional arguments passed to the Series constructor, e.g. ``name``.

    See Also
    --------
    TimeSeriesFrame
    pd.Series
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

    # --------------------------------------------------------------------------
    # time series functionality
    # --------------------------------------------------------------------------
    @property
    def is_timedata(self):
        """
        Is the underlying stored data a time series

        Returns
        -------
        boolean
        """
        return isinstance(self.dtype, TimeDtype)

    @property
    def time_index(self):
        """
        Accessor for the time index of the underlying time index.

        Returns
        -------
        np.ndarray

        Raises
        ------
        TypeError
            if the underlying data is no time series
        """
        if self.is_timedata:
            return self._values.time_index
        else:
            TypeError("The underlying data does not have a time index.")

    @property
    def data(self):
        """
        Accessor to the underlying data object

        Returns
        -------
        np.ndarray
        """
        if self.is_timedata:
            return self._values.data
        return self._values

    def tabularise(self, return_array=False):
        """
        Convert to underlying data into a 2-dimensional table

        Parameters
        ----------
        return_array : boolean, default False
            shall the result be returned as np.ndarray

        Returns
        -------
        table
            if `return_array` is False a pd.DataFrame, else np.ndarray
        """
        if self.is_timedata:
            return self._values.tabularise(self.name, return_array)
        else:
            if return_array:
                return self.to_numpy().reshape(-1, 1)
            else:
                return self

    tabularize = tabularise

    def slice_time(self, time_index, inplace=False):
        """
        Slice a time series across the time axis.

        Parameters
        ----------
        time_index : list, np.ndarray, pd.Index
            indices to be included in the slice
        inplace : boolean, default False
            Not implemented yet

        Returns
        -------
        TimeSeries

        See Also
        --------
        TimeArray.slice_time
        """
        if inplace:
            # TODO: enable inplace slicing
            raise NotImplementedError("inplace slicing of time series is "
                                      "not supported yet")

        return self._constructor(self._values.slice_time(time_index),
                                 index=self.index, name=self.name)

