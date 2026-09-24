#!/usr/bin/env python3
"""
Minimal reproduction case for SubLOF TypeError with timedelta window_size.

Bug: SubLOF crashes with raw TypeError when window_size is timedelta
     but input data has an integer RangeIndex.

Expected: Should raise a clear ValueError explaining the mismatch,
          e.g. "window_size is timedelta but X.index is integer-based."
"""

import numpy as np
import pandas as pd
from datetime import timedelta
from sktime.detection.lof import SubLOF

# Integer-indexed DataFrame (the default when you create a DataFrame)
X = pd.DataFrame({"col_0": np.linspace(1, 10, 10)})

detector = SubLOF(n_neighbors=3, window_size=timedelta(days=5))

print("SubLOF timedelta/RangeIndex mismatch bug reproduction")
print(f"  X.index type : {type(X.index).__name__}")
print(f"  window_size  : {detector.window_size} (timedelta)")
print()

try:
    detector.fit(X)
    print("No error (unexpected)")
except ValueError as e:
    print(f"✓ ValueError (expected): {e}")
except TypeError as e:
    print(f"✗ BUG - TypeError (unexpected): {e}")
    print()
    print("  The error surfaces inside _split_into_intervals() at lof.py:210")
    print("  because 'x_span' is an int but 'interval_size' is a timedelta.")
    print("  There is no guard for the reverse case:")
    print("    isinstance(interval_size, timedelta) and is_integer_index(x)")
    import traceback
    traceback.print_exc()
