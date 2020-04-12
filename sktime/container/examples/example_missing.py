# Script that shows the basic handling of missing data in TimeArrays

import numpy as np
from sktime.container import TimeArray

# Create data without missing data
data = TimeArray(np.array([[1., 2., 3.],
                           [4., 4., 4.],
                           [5., 6., 7.],
                           [8., 8., 8.]]))
print(data.isna())

# Set some data entries to missing
data[1] = TimeArray(np.array([[4., np.nan, 4.]]))
print(data.isna())   # row 1 not considered missing because it contains some data
print(data.hasna())  # but it is flagged in .hasna()

# Set the data of an entire row to missing
data[1] = TimeArray(np.array([[np.nan, np.nan, np.nan]]))
print(data.isna())   # row 1 still not missing because it has valid time index
print(data.hasna())  # still flagged in .hasna()

# Set the data and index of an entire row to missing
data[1] = TimeArray(np.array([[np.nan, np.nan, np.nan]]),
                    time_index=np.array([[np.nan, np.nan, np.nan]]))
print(data.isna())  # Now it is flagged in both
print(data.hasna())

# The same can be achieved by assigning None
data[2] = None
print(data.isna())

# When printing the TimeArray, missing rows are shown as "None"
print(data)

# Under the hood, missing data is represented as all missing values in both the
# data and the time index.
data.data
data.time_index

# This structure is preserved when selecting a smaller section of the TimeArray
# with both missing and observed rows
data[:2].data
data[:2].time_index
print(data[:2])

# If only a single missing row is selected, None is returne
print(data[1])

# If multiple rows are selected that are all missing, a TimeArray with
# width 0 in the underlying time index and data are returned
data[1:3].data
data[1:3].time_index
print(data[1:3])

# Note: slice_time currently does not work with missing indices
data[:2].slice_time([1, 2]).data
data[:2].slice_time([1, 2]).time_index
print(data[:2].slice_time([1, 2]))

