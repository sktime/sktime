# Classifiers, regressors, early classifiers and clusterers

## Classifiers

**Location:** a family sub-package under `sktime/classification/`, such as
`distance_based/`, `dictionary_based/`, `feature_based/`, `interval_based/`,
`kernel_based/`, `shapelet_based/`, `hybrid/`, `deep_learning/` or `ensemble/`. Pick the
family the method belongs to, add a private module such as `_my_classifier.py`, and export
the class from that family's `__init__.py` and `__all__`.

**API reference:** `docs/source/api_reference/classification.rst`, in the family's section.

**Tests:** `sktime/classification/tests/test_all_classifiers.py` runs automatically. Add
specific tests in the family's `tests/` folder.

Mandatory: `_fit(self, X, y)`, `_predict(self, X)`.

Optional: `_predict_proba(self, X)`, `_get_fitted_params(self)`. If `_predict_proba` is
missing the default gives hard 0/1 probabilities, so implement it when the method has
real probabilities.

## Regressors

There's no regressor template. Start from `classification.py`, change the base class to
`BaseRegressor` from `sktime.regression.base`, and drop what only applies to classes
(`_predict_proba`, class labels). `sktime/regression/dummy/_dummy.py` is a small complete
example.

**Location:** a sub-package under `sktime/regression/`.
**API reference:** `docs/source/api_reference/regression.rst`.

## Early classifiers

**Location:** `sktime/classification/early_classification/`.
**API reference:** the early classification section of
`docs/source/api_reference/classification.rst`.

Mandatory: `_fit(self, X, y)`, `_predict(self, X)`, `_update_predict(self, X)`, `_score`.

## Clusterers

**Location:** `sktime/clustering/`, either a module like `dbscan.py` or a sub-package.
**API reference:** `docs/source/api_reference/clustering.rst`.

Mandatory: `_fit(self, X)`.

Then declare what's implemented with `capability:predict`, `capability:predict_proba` and
`capability:out_of_sample`, and implement `_predict` and `_predict_proba` to match.

## Tags to decide on

| Tag | Set it to |
|---|---|
| `capability:multivariate` | `True` if multivariate panels work |
| `capability:unequal_length` | `True` if series of different lengths work |
| `capability:missing_values` | `True` if NaNs work |
| `capability:multithreading` | `True` if `n_jobs` is respected |
| `property:randomness` | `"derandomized"` if a seed makes it reproducible, `"stochastic"` if it never is, `"deterministic"` otherwise |
| `capability:random_state` | `True` if there's a `random_state` parameter |
| `capability:predict_proba` | Classifiers and clusterers: `True` if real probabilities are implemented |
| `y_inner_mtype` | Classifiers, regressors: `"numpy1D"` unless you need otherwise |

## Common review findings

- Don't write to reserved attributes, e.g. `classes_`, `n_classes_` or `fit_time_`. The base
  class sets them and they're listed at the top of the template.
- With a `random_state` parameter, derive every random draw from it. `test_fit_idempotent`
  fits and compares twice, so unseeded randomness makes it flaky.
- Deep learning estimators should keep their architecture code in `sktime/networks/` and
  use small `get_test_params`, e.g. one epoch and few filters. Custom parameters have to
  reach the network: `test_dl_constructor_initializes_deeply` checks this.
- If the method is just a transformer followed by a sklearn classifier, consider whether a
  pipeline or the `sktime.classification.compose` tools already cover it.
