"""Three import shapes, each in a fresh subprocess (no cache cross-contamination)."""
import subprocess
import sys

CHECKS = [
    # 1. leaf from-import (what pickle uses)
    "from sktime.transformations.series.detrend.mstl import MSTL; assert MSTL.__name__ == 'MSTL'",
    # 2. submodule from parent (breaks with naive sys.modules tricks)
    "from sktime.transformations.series.detrend import mstl; assert mstl.MSTL.__name__ == 'MSTL'",
    # 3. re-export from detrend/__init__.py (must be the *same* class)
    "from sktime.transformations.series.detrend import MSTL as A; "
    "from sktime.transformations.series.detrend.mstl import MSTL as B; assert A is B",
]

for code in CHECKS:
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print("OK  " if r.returncode == 0 else "FAIL", code)
    if r.returncode:
        print(r.stderr.rstrip())
