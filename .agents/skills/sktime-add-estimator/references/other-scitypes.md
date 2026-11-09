# Splitters, aligners, distances and kernels, parameter estimators, catalogues

For each of these the template is short and the header lists the required methods. Read
it, and copy from the nearest neighbour in the target folder.

## Splitters

**Template:** `split.py`. **Base:** `BaseSplitter` from `sktime.split.base`.
**Location:** `sktime/split/`, exported in `sktime/split/__init__.py`.
**API reference:** `docs/source/api_reference/split.rst`.

Mandatory: `_split(self, y)`. Optional: `_split_loc(self, y)`, `get_n_splits(self, y)`.

Tags: `split_hierarchical`, `split_series_uses` (`"iloc"` by default).

## Aligners

**Template:** `alignment.py`. **Base:** `BaseAligner` from `sktime.alignment.base`.
**Location:** `sktime/alignment/`.
**API reference:** `docs/source/api_reference/alignment.rst`.

Mandatory: `_fit(self, X, Z)`, `_get_alignment(self)`. Optional: `_get_distance`,
`_get_distance_matrix`.

Tags: `capability:multiple-alignment`, `capability:distance`,
`capability:distance-matrix`, `capability:unequal_length`.

## Distances and kernels

**Templates:** `dist_kern_panel.py` for time series, `dist_kern_tab.py` for tabular data.
**Bases:** `BasePairwiseTransformerPanel`, `BasePairwiseTransformer`.
**Location:** `sktime/dists_kernels/`.
**API reference:** `docs/source/api_reference/dists_kernels.rst`.

Mandatory: `_transform(self, X, X2=None)`. With `X2=None`, compute the matrix of `X` with
itself.

## Parameter estimators

**Template:** `param_est.py`. **Base:** `BaseParamFitter` from `sktime.param_est.base`.
**Location:** `sktime/param_est/`.
**API reference:** `docs/source/api_reference/param_est.rst`.

Mandatory: `_fit(self, X)`. Optional: `_update(self, X)`, `_get_fitted_params(self)`.

Write results to attributes ending in `_` and document them in the `Attributes` section.
They're the estimator's output, so the docstring is the only description users get.

Tags: `scitype:X`, `capability:missing_values`, `capability:multivariate`.

## Catalogues

**Template:** `catalogue.py`. **Base:** `BaseCatalogue` from `sktime.catalogues.base`.
**Location:** `sktime/catalogues/`.
**API reference:** `docs/source/api_reference/catalogues.rst`.

Mandatory: `_get(self)`, returning a mapping from categories to lists of items. Fill in the
`info:` tags with a name, description and source.
