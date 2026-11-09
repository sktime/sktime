# Performance metrics

## Forecasting metrics

**Location:** a private module such as
`sktime/performance_metrics/forecasting/_mb.py`. Export the class from
`sktime/performance_metrics/forecasting/__init__.py`, in both the import list and `__all__`.
`git show 6dee9bf01 --stat` is a good minimal example: one new module, the export, and the
API reference entry.

**API reference:** `docs/source/api_reference/performance_metrics.rst`.

**Naming convention:** `*Score` is a value to maximize. `*Error` or `*Loss` is a value to
minimize. Don't call a signed quantity an error unless it's non-negative: `MeanBias` was
renamed from a "mean error" name for this reason.

### Methods

Implement at least one of these, and preferably both:

- `_evaluate_by_index(self, y_true, y_pred, **kwargs)`: the metric at each time point.
  If it's the only one, `_evaluate` defaults to its arithmetic mean.
- `_evaluate(self, y_true, y_pred, **kwargs)`: the overall value. If it's the only one,
  `_evaluate_by_index` falls back to jackknife pseudo-values, which is slower and less exact.

Inputs arrive as `pd.DataFrame`. `multioutput`, `multilevel` and `by_index` come from the
base class. Check how the `multioutput` argument is respected by copying an existing
metric, such as `_mb.py`, rather than working it out from scratch.

### Common review findings

- **Get the sign and direction right, and test the value.** `check_estimator` can't catch a
  flipped sign. Before writing the formula, state the definition from the literature. Bias
  is predicted minus true. Then add a test on a small example worked out by hand.
- Write the formula in the docstring with `:math:`, and describe what's returned for
  `evaluate` and `evaluate_by_index`, as `MeanBias` does.
- The docstring text and the code must agree on what a positive value means.
- Use `r"""` for docstrings with LaTeX.
- A doctest printing a numpy scalar is version-sensitive (`np.float64(0.05)`). Wrap the
  value in `float(...)` in the example rather than skipping the doctest.

## Detection metrics

**Location:** `sktime/performance_metrics/detection/`.
**API reference:** `docs/source/api_reference/performance_metrics.rst`.

Base class `BaseDetectionMetric`, template `extension_templates/metric_detection.py`.

Mandatory: `_evaluate`, with the signature in the template.

Tags: `scitype:y` (`"points"` or segments), `requires_X`, `requires_y_true`,
`lower_is_better`.
