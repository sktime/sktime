# READY TO FILE AT: https://github.com/sktime/sktime/issues/new?template=bug_report.md
# Title (paste into the title field):
# [BUG] SubLOF crashes with TypeError when window_size is timedelta but X has integer RangeIndex

---

## LLM generated content, by Claude Sonnet 4.6

**Describe the bug**

`SubLOF._fit()` raises a raw `TypeError: unsupported operand type(s) for /: 'int' and
'datetime.timedelta'` when `window_size` is a `datetime.timedelta` but the input
DataFrame has an integer `RangeIndex`. The error message gives no guidance about the
actual type mismatch — the user has no way to know what went wrong.

**To Reproduce**

```python
import numpy as np
import pandas as pd
from datetime import timedelta
from sktime.detection.lof import SubLOF

X = pd.DataFrame({"col_0": np.linspace(1, 10, 10)})  # default integer RangeIndex
detector = SubLOF(n_neighbors=3, window_size=timedelta(days=5))
detector.fit(X)
```

Full traceback:
```
File "sktime/detection/base/_base.py", line 190, in fit
    self._fit(X=X_inner, y=y)
File "sktime/detection/lof.py", line 190, in _fit
    intervals = self._split_into_intervals(X.index, self.window_size)
File "sktime/detection/lof.py", line 210, in _split_into_intervals
    n_intervals = math.floor(x_span / interval_size) + 1
TypeError: unsupported operand type(s) for /: 'int' and 'datetime.timedelta'
```

**Expected behavior**

A clear `ValueError` should be raised before the division is attempted:

```
ValueError: window_size is a timedelta but X.index is integer-based (RangeIndex).
Pass an integer window_size, or provide a DatetimeIndex / PeriodIndex for the input data.
```

**Additional context**

Root cause: `_split_into_intervals()` in `lof.py` (line 208) only handles one
direction of the index/window_size type mismatch:

```python
# existing guard — int window_size + non-integer index → converts interval_size
if isinstance(interval_size, int) and not is_integer_index(x):
    interval_size = x.freq * interval_size
```

The **reverse case** — `timedelta` window_size + integer index — is missing entirely,
so `x_span` (an `int`) is divided by `interval_size` (a `timedelta`), producing the
opaque `TypeError`.

A symmetric guard would fix this:

```python
from datetime import timedelta as _timedelta
if isinstance(interval_size, _timedelta) and is_integer_index(x):
    raise ValueError(
        "window_size is a timedelta but X.index is integer-based. "
        "Use an integer window_size or provide a DatetimeIndex."
    )
```

I am investigating `_split_into_intervals` to confirm this is the only affected site
and will open a PR with the fix shortly.

**Versions**

<details>

```
System:
    python: 3.11.15
executable: /usr/local/bin/python3
   machine: Linux-4.4.0-x86_64-with-glibc2.39

Python dependencies:
          pip: 24.0
       sktime: 0.40.1
      sklearn: 1.7.2
       skbase: 0.13.2
        numpy: 2.3.5
        scipy: 1.17.1
       pandas: 2.3.3
```

</details>
