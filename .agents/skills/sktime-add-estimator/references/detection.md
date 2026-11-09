# Detectors (anomalies, changepoints, segments)

**Location:** a module at `sktime/detection/<name>.py` or a sub-package.

**API reference:** `docs/source/api_reference/detection.rst`.

**Tests:** `sktime/detection/tests/test_all_detectors.py` runs automatically. Add specific
tests as `sktime/detection/tests/test_<name>.py`.

## Methods

Mandatory: `_fit(self, X, y=None)`, `_predict(self, X)`.

Optional: `_update(self, X, y=None)`, plus the score methods described in
`extension_templates/detection.py`.

## Tags to decide on

| Tag | Set it to |
|---|---|
| `task` | `"anomaly_detection"`, `"change_point_detection"` or `"segmentation"`. This fixes what `predict` has to return |
| `learning_type` | `"unsupervised"`, `"semi_supervised"`, or `"supervised"` if `y` labels are used in `fit` |
| `capability:multivariate` | `True` if multivariate series work |
| `capability:missing_values` | `True` if NaNs work. The template still says `capability:missing_data`, but the base class and tag registry use this name |
| `fit_is_empty` | `True` if nothing is learned in `fit`, as with stateless detectors |

Read the docstring of `_predict` in the template for the exact return format per `task`.
Getting this format wrong is the most common failure in `test_all_detectors.py`.

## Common review findings

- `predict` returns locations in the format `task` requires, not a mask or raw scores.
- Don't re-implement windowing or penalties. Look in `sktime/detection/_costs`,
  `_penalties.py`, `_change_scores` and `_anomaly_scores` first.
- If you're adding a metric for detection as well, see `metrics.md`.
