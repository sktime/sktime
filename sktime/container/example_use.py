import numpy as np
import pandas as pd
from sktime.datasets import load_gunpoint
from sktime.container import TimeFrame, TimeSeries

# Load the current nested dataset
X_nested = load_gunpoint(return_X_y=False)

# Convert into a TimeFrame
X = TimeFrame(X_nested)

# Tabularise
X.tabularise()

# Segment
# TODO: current function lives only in TimeArray; this should be passed through in a function of TimeSeries
X.dim_0.values.slice_time([2, 3, 4])
