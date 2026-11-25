# READY TO FILE AT: https://github.com/sktime/sktime/issues/new?template=bug_report.md
# Title (paste into the title field):
# [BUG] DummyRegularAnomalies crashes with IndexError when fit on empty DataFrame

---

## LLM generated content, by Claude Sonnet 4.6

**Describe the bug**

`DummyRegularAnomalies._fit()` raises a raw `IndexError` when passed an empty DataFrame
(zero rows). The error surfaces deep in pandas internals with no informative message.
The detector declares `capability:missing_values: True`, implying robust input handling,
but empty data is not guarded at all.

**To Reproduce**

```python
import pandas as pd
from sktime.detection.dummy import DummyRegularAnomalies

X_empty = pd.DataFrame()

detector = DummyRegularAnomalies()
detector.fit(X_empty)
```

Full traceback:
```
File "sktime/detection/base/_base.py", line 190, in fit
    self._fit(X=X_inner, y=y)
File "sktime/detection/dummy/_dummy_regular_an.py", line 70, in _fit
    self.first_index_ = X_index[0]
IndexError: index 0 is out of bounds for axis 0 with size 0
```

**Expected behavior**

A clear `ValueError` should be raised, for example:

```
ValueError: X cannot be empty. DummyRegularAnomalies requires at least 1 row.
```

**Additional context**

Root cause: `_dummy_regular_an.py` line 70 accesses `X_index[0]` unconditionally.
A one-line guard is all that is needed:

```python
# sktime/detection/dummy/_dummy_regular_an.py  _fit()
if len(X_index) == 0:
    raise ValueError("X cannot be empty; DummyRegularAnomalies requires at least 1 row.")
self.first_index_ = X_index[0]
```

I am investigating the `_fit` method to confirm this is the only affected site
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
