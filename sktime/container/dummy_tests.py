import pandas as pd
import numpy as np
import datetime

import pandas._testing as tm

from sktime.datasets import load_gunpoint

from extensionarray.timeframe import TimeFrame
from extensionarray.array import TimeArray

d_scalar = {'x': [1, 2], 'y': ['a', 'b']}
d_ar1d = {'x': np.array([1, 2]), 'y': ['a', 'b']}
d_ar2d = {'x': np.array([[1, 2], [3, 4]]), 'y': ['a', 'b']}
l_scalar = [[1, 'a'], [2, 'b']]

df = pd.DataFrame(data=d_scalar)
ar0d = np.array(1)
ar1d = np.array([1, 2, 3])
ar2d = np.array([[1, 2, 3], [4, 5, 6]])
ar3d = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])





# ----------------------------------------------------------------------------------------------------------------------
# Tests: TimeFrame
# ----------------------------------------------------------------------------------------------------------------------

# Constructors ---------------------------------------------------------------------------------------------------------

# Constructor: dict with scalars
tf = TimeFrame(data=d_scalar, index=[1])


# Constructor: list with scalars
tf = TimeFrame(data=l_scalar)


# Column: integer
tf = TimeFrame(data=ar2d, columns=[5, 8, 2])

# Column: character
tf = TimeFrame(data=ar2d, columns=['a', 'b', 'c'])
tf = TimeFrame(data=ar3d, columns=['a', 'b'])


# Index: integer
tf = TimeFrame(data=ar3d, index=[10, 20])

# Index: character
tf = TimeFrame(data=ar3d, index=['a', 'b'])



date_index = pd.to_datetime(['1/1/2018', np.datetime64('2018-01-02'),  datetime.datetime(2018, 1, 3)])

# Slicing --------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import datetime

from sktime.datasets import load_gunpoint
from collections import abc

from extensionarray.timeframe import TimeFrame
from extensionarray.timeseries import TimeSeries
from extensionarray.array import TimeArray


X = load_gunpoint()
tf = TimeFrame(X, copy=True)


# __getitem__ indexing
type(tf['dim_0'])
type(tf[['dim_0']])
type(tf[['dim_0', 'class_val']])
# type(tf[['dim_0', 'class_val', 'nonexistent']]) Fail

# .loc indexing
tf.loc[:, 'dim_0'].equals(tf['dim_0'])
tf.loc[:, ['dim_0']].equals(tf[['dim_0']])
isinstance(tf.loc[0:10, 'dim_0'], TimeSeries)
isinstance(tf.loc[0:10, ['dim_0']], TimeFrame)
tf.loc[0:10, ['dim_0']].dim_0.time_index

tf.loc[tf['class_val']==2, :]

# .iloc indexing
tf.iloc[:, 0].equals(tf['dim_0'])
tf.iloc[0, 0]
tf.iloc[[0], 0]
tf.iloc[0, [0]] # TODO: this is due to how pandas selects, which passes the entire row to the TimeArray constructor. How do we deal with this?


# TimeArray slicing


df = pd.DataFrame({'A': ['a', 'b', 'c', 'a']})
df['B'] = df['A'].astype('category')


# ----------------------------------------------------------------------------------------------------------------------
# Tests: TimeArray
# ----------------------------------------------------------------------------------------------------------------------

import pandas as pd
import numpy as np
from extensionarray.array import TimeArray

df = pd.DataFrame(data=[[1, 2], [3, 4]])
ar0d = np.array(1)
ar1d = np.array([1, 2, 3])
ar2d = np.array([[1, 2, 3], [4, 5, 6]])
ar3d = np.array([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]])
d_correct = {'data': df.to_numpy(), 'time_index': [0, 1]}
d_error1 = {'data': df.to_numpy(), 'time_index': [0, 1], 'random': 1}
d_error2 = {'data': df.to_numpy()}
l_num = [[1, 2, 3], [4, 5, 6]]
l_log = [[True, True, False], [False, False, False]]
l_str = [['a', 'b', 'c'], ['d', 'e', 'f']]
l_uneven = [[1, 2], [4, 5, 6]]

# Constructors ---------------------------------------------------------------------------------------------------------

# Constructor: Empty
ta = TimeArray() # Error

# Constructor: None
ta = TimeArray(data=None)

# Constructor: DataFrame
ta = TimeArray(df)

# Constructor: TimeArray
ta = TimeArray(TimeArray(df))

# Constructor: Series
ta = TimeArray(df.iloc[:, 0])

# Constructor: dict
ta = TimeArray(d_correct)
np.all(TimeArray(d_correct, time_index=[1, 2]).time_index == np.array([1, 2]))
ta = TimeArray(d_error1)
ta = TimeArray(d_error2)

# Constructor: list
ta = TimeArray(l_num)
ta = TimeArray(l_log)
ta = TimeArray(l_str)
ta = TimeArray(l_uneven)

# Constructor: np.ndarray (0D)
ta = TimeArray(ar0d)

# Constructor: np.ndarray (1D)
ta = TimeArray(ar1d)

# Constructor: np.ndarray (2D)
ta = TimeArray(ar2d)

# Constructor: np.ndarray (3D)
ta = TimeArray(ar3d)

#