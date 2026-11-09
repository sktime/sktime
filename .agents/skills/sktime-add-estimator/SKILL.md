---
name: sktime-add-estimator
description: Add a new estimator to sktime, end to end, for contributors. Covers choosing the scitype and extension template, tags, get_test_params, docstrings, the API reference entry, and check_estimator. Use this whenever someone wants to add, implement, port, wrap or interface a forecaster, transformer, classifier, regressor, clusterer, detector, metric, splitter, aligner, distance, kernel or parameter estimator in sktime, even if they only name the algorithm or paper and never say "estimator". For pretrained time series foundation models, use sktime-add-foundation-model instead.
---

# Adding an estimator to sktime

The person using this is often a first-time sktime contributor. Do the work, but
explain the sktime-specific decisions as you make them, so they can defend the PR in
review. Don't commit, push or open a PR unless they ask.

## 1. Check it should be added

1. Look for an existing implementation, since duplicates are the most common reason a
   new-estimator PR gets closed:

   ```bash
   grep -rli "<algorithm name>" sktime docs/source/api_reference
   gh pr list --repo sktime/sktime --state open --search "<algorithm name>"
   gh issue list --repo sktime/sktime --state open --search "<algorithm name>"
   ```

   If an open PR or an assigned issue covers it, stop and tell the user. They should
   coordinate there instead of opening a competing PR.
2. Establish what is being added: a native implementation, or an interface to a third
   party package. An interface means a soft dependency (step 4). If the package is
   compiled with cython, read "Estimators dependent on cython" in
   `docs/source/developer_guide/add_estimators.rst` first.
3. If it wraps pretrained deep learning weights, switch to `sktime-add-foundation-model`.
4. Get the paper or canonical reference. The docstring needs it, and a reference
   implementation gives you something to test numbers against.

## 2. Pick the scitype and template

The templates in `extension_templates/` are authoritative for each method's contract.
Read the template, and the reference file for the scitype, before writing code.

| Adding a... | Template | Base class | Reference |
|---|---|---|---|
| Forecaster, point only and simple | `forecasting_supersimple.py` or `forecasting_simple.py` | `BaseForecaster` | `references/forecasting.md` |
| Forecaster, probabilistic, composite or with `update` | `forecasting.py` | `BaseForecaster` | `references/forecasting.md` |
| Transformer, series to series | `transformer_supersimple.py` or `transformer_simple.py` | `BaseTransformer` | `references/transformations.md` |
| Transformer, series to features | `transformer_supersimple_features.py` | `BaseTransformer` | `references/transformations.md` |
| Transformer, anything more complex | `transformer.py` | `BaseTransformer` | `references/transformations.md` |
| Time series classifier | `classification.py` | `BaseClassifier` | `references/classification-regression-clustering.md` |
| Time series regressor | `classification.py`, adapted | `BaseRegressor` | `references/classification-regression-clustering.md` |
| Early classifier | `early_classification.py` | `BaseEarlyClassifier` | `references/classification-regression-clustering.md` |
| Clusterer | `clustering.py` | `BaseClusterer` | `references/classification-regression-clustering.md` |
| Anomaly, changepoint or segment detector | `detection.py` | `BaseDetector` | `references/detection.md` |
| Forecasting metric | `metric_forecasting.py` | `BaseForecastingErrorMetric` | `references/metrics.md` |
| Hierarchy-aware forecasting metric | `metric_forecasting_hierarchical.py` | `BaseForecastingErrorMetric` | `references/metrics.md` |
| Detection metric | `metric_detection.py` | `BaseDetectionMetric` | `references/metrics.md` |
| Splitter | `split.py` | `BaseSplitter` | `references/other-scitypes.md` |
| Aligner | `alignment.py` | `BaseAligner` | `references/other-scitypes.md` |
| Distance or kernel, panel | `dist_kern_panel.py` | `BasePairwiseTransformerPanel` | `references/other-scitypes.md` |
| Distance or kernel, tabular | `dist_kern_tab.py` | `BasePairwiseTransformer` | `references/other-scitypes.md` |
| Parameter estimator | `param_est.py` | `BaseParamFitter` | `references/other-scitypes.md` |
| Catalogue of estimators and datasets | `catalogue.py` | `BaseCatalogue` | `references/other-scitypes.md` |

Choose the simplest template that covers what the estimator does. Before picking one, check
it isn't really a different scitype: "feature extraction plus a classifier" is usually
a pipeline of existing estimators, not a new one.

## 3. Implement

Copy the template to the target location from the reference file, and work through every
`todo`. When you finish, none of the template's `todo` comments or usage header should
remain, because a leftover `todo` reads as unfinished work and the docstring is published.

These rules apply to every scitype, and the test suite enforces most of them:

- **Implement the private methods** (`_fit`, `_predict`, `_transform`, ...). The public
  `fit`, `predict` and so on live in the base class, which handles input checks, mtype
  conversion, vectorization and state. Don't override them.
- **`__init__` only stores parameters.** Write each one to `self` under the same name,
  unchanged, then call `super().__init__()`. Validation or coercion goes to a private copy
  or to `__post_init__`, because `get_params`, `clone` and `set_params` rely on the stored
  values being exactly what was passed in.
- **Fitted state goes in attributes ending in `_`.** Don't overwrite hyper-parameters in
  `_fit`. Don't write to the reserved variables listed at the top of the template.
- **Prefer the base class to your own checks.** The inner mtype tags guarantee the input
  format, so `isinstance` and `ndim` branches on inputs are dead code. Reviewers will ask
  you to remove them.
- **Keep soft dependency imports out of module level and `__init__`.** See step 4.

## 4. Tags

Set tags so they describe what the code really does. Tests read them to decide what to
exercise, so a wrong capability tag either fails or hides behaviour from the tests.

Every scitype needs:

```python
_tags = {
    "authors": ["their-github-handle"],      # plus upstream authors, for interfaces
    "maintainers": ["their-github-handle"],  # or drop it if nobody will own it
    # plus the scitype and capability tags from the template
}
```

- Use real GitHub handles. `"sktime developers"` and organization names don't belong here.
- When interfacing a third party algorithm, keep its authors in `authors` as well. Mirror
  the list in the module's `__author__`.
- Look up a tag's meaning with
  `python -c "from sktime.registry import all_tags; print(all_tags(estimator_types='forecaster', as_dataframe=True))"`
  rather than guessing.

**If the estimator has a soft dependency:**

- Import it inside methods, or in `__post_init__`, never at module top level or in
  `__init__`. Otherwise `import sktime` fails for users without the package.
- Set `"python_dependencies": ["pkg>=1.2"]` as a PEP 440 specifier. The base class then
  raises a useful error for you, so don't add your own try/except or warning.
- Set `"tests:vm": True`. The dependency guide requires it, and it gives the estimator its
  own CI job with those dependencies installed.
- Don't add a one-estimator dependency to `pyproject.toml` extras. The VM job installs it
  from the tag. Read `docs/source/developer_guide/dependencies.rst` if it seems necessary.
- If it has its own test file, add `"tests:specific": ["sktime.<path>.tests.test_<name>"]`.
  Without that tag the VM job never runs the file, and CI stays green.

## 5. `get_test_params`

`check_estimator` builds its test instances from this method.

- Return **at least two** parameter sets. `test_get_test_params_coverage` fails on one.
- Make them differ in ways that hit different code paths, rather than one changed number.
- Keep them small enough that the whole suite runs in seconds on CI: few iterations, small
  networks, short windows.
- Import anything it needs (estimators, soft dependencies) inside the method.

## 6. Docstring and API reference

Reviewers check these first, and most repeat comments on new estimators come from here.

- Follow numpydoc: one-line summary, extended summary, then `Parameters`, `Attributes`,
  `See Also`, `References`, `Examples`.
- **Every `__init__` parameter is documented**, in the form `name : type, default=value`.
  The spaces around the colon matter. Write what it does and what values are valid.
- **`Examples` needs a real `>>>` example.** `test_class_has_doctest_example` fails without
  one, and `test_doctest_examples` runs it, so it has to work. Use a sktime dataset, and
  mark anything slow or needing a download with `# doctest: +SKIP`.
- Cite the paper in `References` as `.. [1] ...`, and refer to it with `[1]_`.
- **Add the class to `docs/source/api_reference/<task>.rst`**, under the matching section
  and `currentmodule`, following the entries around it. No test catches a missing entry,
  so it is easy to forget, and maintainers end up adding it for you.
- Export it from the package's `__init__.py` and `__all__` if neighbouring estimators are
  exported that way. The reference file says when that applies.

## 7. Check the numbers, not only the interface

`check_estimator` proves the estimator follows the API. It doesn't prove it computes the
right thing. A merged metric once needed a maintainer to fix its sign after every
interface test passed. So add at least one test with hand-computed or reference values
in the module's `tests/` folder. When a reference implementation exists, generate expected
values from it with the `sktime-golden-tests` skill. Gate tests with soft dependencies:

```python
@pytest.mark.skipif(
    not run_test_for_class(MyEstimator),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
```

`run_test_for_class` comes from `sktime.tests.test_switch`.

## 8. Verify

Run these and report the real result. A skipped test isn't a passed one.

```bash
python -c "
from sktime.utils.estimator_checks import check_estimator
from sktime.<path> import MyEstimator
results = check_estimator(MyEstimator)
failed = {k: v for k, v in results.items() if v != 'PASSED'}
print(len(results), 'run,', len(failed), 'failed')
print(failed)
"
pre-commit run --files <changed files>
```

If something fails, narrow it down with `tests_to_run="test_name"` or
`fixtures_to_run="test_name[MyEstimator-0]"`, then add `raise_exceptions=True` to get the
traceback.

To run the estimator's test file, the flags depend on the `tests:vm` tag, because
`setup.cfg` turns on change detection and `run_test_for_class` skips VM estimators in a
normal run:

```bash
# no tests:vm tag
pytest sktime/<path>/tests/test_<name>.py --only_changed_modules False --matrixdesign False -n0
# tests:vm: True, this is what the CI VM job runs
pytest sktime/<path>/tests/test_<name>.py --only_vm_estimators True --only_changed_modules False --matrixdesign False -n0
```

Look at the pass count. "0 selected" or all skipped means nothing ran. Use `-rs` to see
why. Common reasons are `False_requires_vm` (use the second command) and
`False_required_deps_missing` (install the dependency).

If a test genuinely can't pass, ask the user before skipping it. `tests:skip_by_name` needs
a comment next to it explaining why, ideally linking an issue. Avoid `tests:skip_all`,
which disables every test for the estimator.

## 9. Hand over

Tell the user what you added, the verification output, and anything not verified. Then
suggest running `sktime-pr-check` before opening the PR. New contributors also need to add
themselves to `.all-contributorsrc`, which that skill covers.
