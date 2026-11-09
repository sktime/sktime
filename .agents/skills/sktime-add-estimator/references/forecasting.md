# Forecasters

**Location:** a single module, `sktime/forecasting/<name>.py`, with `__all__`. Most
forecasters are imported from their module, e.g. `from sktime.forecasting.theta import
ThetaForecaster`. Check neighbouring modules before adding to an `__init__.py`.

**API reference:** `docs/source/api_reference/forecasting.rst`, in the section that fits,
under a `.. currentmodule:: sktime.forecasting.<name>` line.

**Tests:** `sktime/forecasting/tests/test_<name>.py`. Interface tests live in
`sktime/forecasting/tests/test_all_forecasters.py` and run automatically.

## Methods

Mandatory: `_fit(self, y, X, fh)`, `_predict(self, fh, X)`.

Optional, in `extension_templates/forecasting.py`:

- `_update(self, y, X=None, update_params=True)`
- `_predict_quantiles(self, fh, X, alpha)`, `_predict_interval(self, fh, X, coverage)`,
  `_predict_var(self, fh, X, cov)`, `_predict_proba(self, fh, X, marginal)`
- `_get_fitted_params(self)`

Keep the standard signatures even if a static checker complains about them.

## Tags to decide on

| Tag | Set it to |
|---|---|
| `y_inner_mtype` | The format you want in `_fit`, usually `"pd.Series"` or `"pd.DataFrame"`. Add panel or hierarchical mtypes only if the method handles them natively, otherwise sktime vectorizes for you |
| `X_inner_mtype` | Same, for exogenous data |
| `capability:multivariate` | `True` only if the method models several variables natively. With `False`, sktime fits one copy per variable |
| `capability:exogenous` | `True` only if `X` is used. With `False`, inner methods always see `X=None` |
| `requires-fh-in-fit` | `True` if the model needs the horizon when fitting, as direct reduction does |
| `capability:insample` | `False` if it can't predict in-sample |
| `capability:pred_int` | `True` if it implements any probabilistic method |
| `capability:missing_values` | `True` only if `y` with NaNs works |
| `capability:update` | `True` if `_update` is implemented natively |

## Common review findings

- **Don't read `self._y` or `self._X` in `_predict`.** The `remember_data` default is moving
  to `False` in 1.3.0, and then those attributes are not stored. If predict needs history,
  keep your own copy in `_fit` and extend it in `_update`, using `update_data` from
  `sktime.datatypes`. A forecaster that stores context needs an `_update`, or `update` will
  move the cutoff while `predict` uses stale data.
- **`fh` can be anything the user passes.** Non-contiguous horizons such as `[2, 5]` and
  in-sample steps are valid. Convert with `fh.to_relative(self.cutoff)`, compute up to the
  max step, select the requested steps, and index the result with
  `fh.to_absolute(self.cutoff)`.
- **Return the same mtype you received**, with the original variable name and an index
  matching `fh`.
- **Quantiles need more than `_predict_quantiles`.** The default `_predict_proba` assumes a
  normal distribution, so a method that emits quantiles should also provide its own
  `_predict_proba`, and the two have to agree.
