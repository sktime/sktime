# Script that shows the basic functionality of TimeFrame based on the gunpoint
# example dataset

from sktime.datasets import load_gunpoint
from sktime.container import TimeFrame, TimeSeries

# Load the current nested dataset
X_nested = load_gunpoint(return_X_y=False)

# Convert into a TimeFrame
X = TimeFrame(X_nested)

# Tabularise
X.tabularise()

# Segment
X.dim_0.slice_time([2, 3, 4])

