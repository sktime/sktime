# Bug Report: DummyRegularAnomalies Crashes on Empty Data

## Summary
`DummyRegularAnomalies` detector crashes with `IndexError` when given an empty DataFrame, instead of raising a informative error message.

## Bug Description
The detector attempts to access the first index of an empty DataFrame without validation, causing an unhandled IndexError deep in pandas internals.

## To Reproduce
```python
import pandas as pd
from sktime.detection.dummy import DummyRegularAnomalies

# Create empty DataFrame
X_empty = pd.DataFrame()

# This crashes with IndexError instead of raising ValueError
detector = DummyRegularAnomalies()
detector.fit(X_empty)
```

## Expected Behavior
Should raise a clear `ValueError` explaining that input data cannot be empty, such as:
```
ValueError: X cannot be empty. X must have at least 1 row.
```

## Actual Behavior
Raises `IndexError: index 0 is out of bounds for axis 0 with size 0` from pandas internals:
```
File "/home/user/sktime/sktime/detection/dummy/_dummy_regular_an.py", line 70, in _fit
    self.first_index_ = X_index[0]
```

## Root Cause
In `/sktime/detection/dummy/_dummy_regular_an.py` at line 70, the `_fit` method accesses `X_index[0]` without checking if the index is empty.

## Proposed Fix
Add a validation check in the `_fit` method:
```python
def _fit(self, X, y=None):
    X_index = X.index
    
    # Add this validation
    if len(X_index) == 0:
        raise ValueError("X cannot be empty. X must have at least 1 row.")
    
    if isinstance(X_index, pd.DatetimeIndex):
        X_index = pd.PeriodIndex(X_index)
    self.first_index_ = X_index[0]
    return self
```

## Additional Notes
- This is a common edge case that should be handled gracefully by all estimators
- Similar validation should be checked in other detectors (SubLOF, etc.) to ensure consistent behavior
- The detector has tag `"capability:missing_values": True` which suggests it should handle edge cases robustly

## File References
- Affected file: `/sktime/detection/dummy/_dummy_regular_an.py`
- Method: `DummyRegularAnomalies._fit()` (line 47-71)
