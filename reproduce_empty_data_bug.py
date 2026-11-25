#!/usr/bin/env python3
"""
Minimal reproduction case for DummyRegularAnomalies empty data bug.

Bug: DummyRegularAnomalies crashes with IndexError when given empty DataFrame
Expected: Should raise ValueError explaining that data is empty
"""

import pandas as pd
from sktime.detection.dummy import DummyRegularAnomalies

# Create empty DataFrame (0 rows, 0 columns)
X_empty = pd.DataFrame()

# Try to fit the detector
try:
    detector = DummyRegularAnomalies()
    detector.fit(X_empty)
    print("✓ No error (detector handled empty data)")
except ValueError as e:
    print(f"✓ ValueError (expected): {e}")
except IndexError as e:
    print(f"✗ BUG FOUND - IndexError (unexpected): {e}")
    print(f"  Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
