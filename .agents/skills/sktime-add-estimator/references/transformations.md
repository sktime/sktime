# Transformers

**Location:** usually a module at `sktime/transformations/<name>.py`. Larger families have
sub-packages, e.g. `sktime/transformations/detrend/`. Look at where similar transformers
live before choosing.

**API reference:** `docs/source/api_reference/transformations.rst`, in the section matching
what the transformer does.

**Tests:** `sktime/transformations/tests/`.

## Methods

Mandatory: `_transform(self, X, y=None)`.

Optional: `_fit(self, X, y=None)`, `_inverse_transform(self, X, y=None)`, `_update`.

If the transformer learns nothing, set `"fit_is_empty": True` and leave `_fit` out, so that
`fit` becomes a no-op.

## Tags to decide on

| Tag | Set it to |
|---|---|
| `scitype:transform-input` | `"Series"` or `"Panel"`, the input unit the transform works on |
| `scitype:transform-output` | `"Series"`, `"Primitives"` (one row of features per series) or `"Panel"` |
| `scitype:instancewise` | `True` if each series is transformed independently of the others |
| `X_inner_mtype` | The format you want in `_transform` |
| `y_inner_mtype` | `"None"` unless `y` is used |
| `capability:multivariate` | `True` if multivariate input works natively |
| `requires_y` | `True` if `fit` needs `y` |
| `fit_is_empty` | `True` if nothing is learned in `fit` |
| `capability:inverse_transform` | `True` if `_inverse_transform` is implemented |
| `capability:inverse_transform:exact` | `False` if the inverse only approximates |
| `capability:unequal_length` | Whether unequal length panels work |
| `capability:missing_values` | Whether NaNs in the input work |
| `capability:missing_values:removes` | `True` if the output is guaranteed to have no NaNs |

The input and output scitypes decide what the transformer can be used for. A
`"Series"` to `"Primitives"` transformer is a feature extractor, for instance, and sktime
vectorizes over panels and hierarchies for you.

## Common review findings

- Use the `_supersimple` or `_simple` template unless you need more. The full template
  exposes options most transformers don't use.
- Don't mutate the input `X`. Work on a copy.
- If `_inverse_transform` exists, test that inverting a transform recovers the input.
- Keep column names and index through the transform where that makes sense, since pipelines
  depend on them.
