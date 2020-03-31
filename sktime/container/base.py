import numpy as np

# TODO: look into providing indexing for TimeBase

class TimeBase(object):
    def __init__(self, data, time_index):
        if data.ndim > 2 or time_index.ndim > 2:
            raise ValueError("TimeBase can only handle two-dimensional inputs.")

        data = ensure_2d(data)
        time_index = ensure_2d(time_index)
        check_data_index(data, time_index)

        self.data = data
        self.time_index = time_index

    def __eq__(self, other):
        if isinstance(other, TimeBase):
            return (np.all(self.data == other.data)) & (np.all(self.time_index == other.time_index))
        else:
            raise TypeError(f"Cannot compare TimeBase objec to {type(other)}")

    def __add__(self, o):
        if not isinstance(o, TimeBase):
            return NotImplemented

        if np.all(self.time_index != o.time_index):
            raise ValueError("The time indices of two TimeArrays that should be added must be identical.")

        return TimeBase(self.data + o.data, self.time_index)

    def _create_repr(self, length):
        print_len = min(self.data.shape[1], length)
        list_repr = [f"{self.time_index[0, i]}: {self.data[0, i]}" for i in range(print_len)]

        if print_len != self.data.shape[1]:
            list_repr.append("...")

        return "[" + ", ".join(list_repr) + "]"

    def __str__(self):
        return str(self._create_repr(2))

    def __repr__(self):
        return f"ts (n={self.data.shape[1]}): {self._create_repr(5)}"

    def isna(self):
        return np.all(np.isnan(self.data)) & np.all(np.isnan(self.time_index))


# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def check_data_index(data, time_index):
    if data.shape != time_index.shape:
        raise ValueError(f"The shape of data {data.shape} and time index "
                         f"{time_index.shape} differ, must be equal.")

def ensure_2d(arr):
    if arr.ndim == 0:
        return arr[np.newaxis, np.newaxis]
    elif arr.ndim == 1:
        return arr[np.newaxis, :]
    elif arr.ndim == 2:
        return arr
    else:
        raise ValueError(f"A maximum of two dimensions allowed, got {arr.ndim}.")