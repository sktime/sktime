import numpy as np
import pandas as pd


# TODO: look into providing indexing for TimeBase
class TimeBase(object):
    """
    Basic container for a single timeseries made up of key/value pairs.

    Implementation note: this class is required by TimeArray due to
    pandas.ExtensionArray's need for a base object (i.e. a single cell entry)
    that is not a TimeArray, a numpy.ndarray, or a pandas.Series. In the
    nested pandas.Dataframe implementation in sktime this is equivalent to a
    single pandas.Series.

    Parameters
    ----------
    data : TimeArray or ndarray
        The measurements at certain time points.
    time_index : ndarrays
        A time index for each entry in 'data'. Must be of the same shape as
        data.
    """
    def __init__(self, data, time_index):
        if data.ndim > 2 or time_index.ndim > 2:
            raise ValueError("TimeBase can only handle two-dimensional inputs.")

        data = ensure_2d(data)
        time_index = ensure_2d(time_index)
        check_data_index(data, time_index)

        self.data = data
        self.time_index = time_index

    def __eq__(self, other):
        """ Return self==other. """
        if isinstance(other, TimeBase):
            # TimeBases are equal if both data and index are equal
            return (np.all(self.data == other.data)) & \
                   (np.all(self.time_index == other.time_index))
        else:
            raise TypeError(f"Cannot compare TimeBase objec to {type(other)}")

    def __add__(self, o):
        """ Return self+o. """
        if not isinstance(o, TimeBase):
            return NotImplemented

        if np.all(self.time_index != o.time_index):
            # Can only add TimeBases that have values recorded at the same
            # time indices, otherwise addition is not defined
            raise ValueError("The time indices of two TimeArrays that should "
                             "be added must be identical.")

        return TimeBase(self.data + o.data, self.time_index)

    def _create_repr(self, length):
        """
        Create a string representation of the TimeBase object for printing the
        object in the console.

        Parameters
        ----------
        length : int
            maximum number of entries (i.e. key/value pairs) to be included in
            the string

        Returns
        -------
        str
        """
        print_len = min(self.data.shape[1], length)
        list_repr = [f"{self.time_index[0, i]}: {self.data[0, i]}"
                     for i in range(print_len)]

        if print_len != self.data.shape[1]:
            list_repr.append("...")

        return "[" + ", ".join(list_repr) + "]"

    def __str__(self):
        """ Return str(self). """
        return str(self._create_repr(2))

    def __repr__(self):
        """ Return repr(self). """
        return f"ts (n={self.data.shape[1]}): {self._create_repr(5)}"

    def isna(self):
        """
        Detect missing values

        Missing values (i.e. missing information on all data and indices) are
        detected.

        Returns
        -------
        a boolean array of whether my data and indices are all null
        """
        return np.all(np.isnan(self.data)) & np.all(np.isnan(self.time_index))

    def to_series(self):
        """
        Convert to pandas.Series

        Returns
        -------
        pd.Series
        """
        return pd.Series(self.data[0], self.time_index[0])

    def to_numpy(self):
        """
        Convert to Tuple of np.ndarrays

        Returns
        -------
        (data, time_index) : (np.ndarray, np.ndarray)
        """
        return self.data[0], self.time_index[0]


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def check_data_index(data, time_index):
    """
    Check that the time index as a shape compatible with the data.

    Raises ValueError if shapes are different.

    Parameters
    ----------
    data : np.ndarray
        array of values
    time_index : np.ndarray
        time index for each value in `data`

    Returns
    -------
    None
    """
    if data.shape != time_index.shape:
        raise ValueError(f"The shape of data {data.shape} and time index "
                         f"{time_index.shape} differ, must be equal.")


def ensure_2d(arr):
    """
    Force array into 2D shape if <2D or throw error if >2D

    Parameters
    ----------
    arr : np.ndarray

    Returns
    -------
    arr2d : np.ndarray
        input expanded to 2D; if input was <2D, output will be of shape 1xD
    """
    if arr.ndim == 0:
        return arr[np.newaxis, np.newaxis]
    elif arr.ndim == 1:
        return arr[np.newaxis, :]
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"A maximum of 2 dimensions allowed, got {arr.ndim}.")
