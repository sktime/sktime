---
name: sktime-add-estimator
description: Add a new estimator to sktime, end to end, for contributors. Covers choosing the scitype and extension template, tags, get_test_params, docstrings, the API reference entry, and check_estimator. Use this whenever someone wants to add, implement, port, wrap or interface a forecaster, transformer, classifier, regressor, clusterer, detector, metric, splitter, aligner, distance, kernel or parameter estimator in sktime, even if they only name the algorithm or paper and never say "estimator". For pretrained time series foundation models, use sktime-add-foundation-model instead.
---

# Adding an estimator to sktime

The person using this is often a first-time sktime contributor. Do the work, but
explain the sktime-specific decisions as you make them, so they can defend the PR in
review. Don't commit, push or open a PR unless they ask.

The sources for this work are in the repo. Read them rather than working from memory:

- `docs/source/developer_guide/add_estimators.rst`: choosing the scitype, how to fill in a
  template, `check_estimator`, and what adding to sktime itself requires.
- `extension_templates/`: one template per scitype. Each one's header and `todo` comments
  give the base class, the mandatory and optional methods, the tags and the input guarantees
  for the inner methods. They're authoritative for the method contracts.

This skill adds what those leave out: the decisions reviewers push back on, and how to check
the result.

## 1. Check it should be added

1. Run the duplicate check in "Before starting new work" in `AGENTS.md`, with the
   algorithm's name.
2. Establish what is being added: a native implementation, or an interface to a third
   party package. An interface means a soft dependency (step 4). If the package is
   compiled with cython, read "Estimators dependent on cython" in the guide first.
3. If it wraps pretrained deep learning weights, switch to `sktime-add-foundation-model`.
4. Get the paper or canonical reference. The docstring needs it, and a reference
   implementation gives you something to test numbers against.

## 2. Pick the scitype and template

Read "What is my learning task?" in the guide, then list `extension_templates/`. The file
names give the scitype. Read the header of each candidate template before choosing.

- **Choose the simplest variant** that covers what the estimator does. Forecasters and
  transformers have `_supersimple` and `_simple` templates. Use the full one only for what
  they leave out, such as probabilistic forecasts, `update` or composites. The full templates
  expose options most estimators don't need.
- **Check it isn't really a different scitype,** or a composition of existing ones. "Feature
  extraction plus a classifier" is usually a pipeline, not a new estimator. Look at the
  `compose` module of the target scitype before writing one.
- **No template** exists for a scitype that has a base class, such as time series
  regressors: start from the closest template (`classification.py` for a regressor), change
  the base class, drop what doesn't apply, and compare with a small existing estimator of
  that scitype (e.g. `sktime/regression/dummy/_dummy.py`).

## 3. Place it and implement

**Location, export and API entry: copy the nearest neighbour.** Find the existing estimator
most like the new one, then mirror everything about how it's placed:

```bash
grep -rn "class <Neighbour>" sktime
grep -rn "<Neighbour>" sktime/<module>/__init__.py sktime/<module>/*/__init__.py docs/source/api_reference
```

That gives the module or sub-package to put the file in, whether the class is exported from
an `__init__.py` and `__all__` or imported from its own module, and the `api_reference` file,
section and `currentmodule` its entry goes under.

Copy the template there, and work through every `todo`, following "How to use sktime
extension templates" in the guide. When you finish, none of the template's `todo` comments
or usage header should remain, because a leftover `todo` reads as unfinished work and the
docstring is published.

On top of the caveats in the guide, reviewers check:

- **Implement only the private methods** (`_fit`, `_predict`, `_transform`, ...). The base
  class handles input checks, mtype conversion, vectorization and state.
- **Validation or coercion of parameters** goes to a private copy or to `__post_init__`,
  since `__init__` stores them unchanged.
- **Fitted state goes in attributes ending in `_`.** Don't write to the reserved attributes
  listed at the top of the template.
- **Don't re-check what the base class guarantees.** The inner mtype tags fix the input
  format, so `isinstance` and `ndim` branches on inputs are dead code, and reviewers will ask
  you to remove them.
- **Keep soft dependency imports out of module level and `__init__`.** See step 4.

Then check the [review findings](#review-findings-by-scitype) for the scitype.

## 4. Tags

Set tags so they describe what the code really does. Tests read them to decide what to
exercise, so a wrong capability tag either fails or hides behaviour from the tests. The
template lists the tags for the scitype. Look up what a tag means in the registry rather than
guessing:

```bash
python -c "from sktime.registry import all_tags; print(all_tags(estimator_types='<scitype>', as_dataframe=True))"
```

Where the template and the registry disagree on a tag's name, the registry wins.

`authors` and `maintainers` (see the guide) hold real GitHub handles. `"sktime developers"`
and organization names don't belong there. When interfacing a third party algorithm, keep
its authors in `authors` as well, and mirror the list in the module's `__author__`.

**If the estimator has a soft dependency,** follow "Soft dependencies" in `AGENTS.md`. It
covers the import location and the `python_dependencies`, `tests:vm` and `tests:specific`
tags. An import at module top level or in `__init__` makes `import sktime` fail for users
without the package.

## 5. `get_test_params`

`check_estimator` builds its test instances from this method.

- Return **at least two** parameter sets. `test_get_test_params_coverage` fails on one.
- Make them differ in ways that hit different code paths, rather than one changed number.
- Keep them small enough that the whole suite runs in seconds on CI: few iterations, small
  networks, short windows.
- Import anything it needs (estimators, soft dependencies) inside the method.

## 6. Docstring and API reference

Reviewers check these first, and most repeat comments on new estimators come from here.
`docs/source/developer_guide/documentation.rst` has the docstring conventions.

- **Every `__init__` parameter is documented**, in the form `name : type, default=value`.
  The spaces around the colon matter. Write what it does and what values are valid.
- **Public fitted attributes** are documented under `Attributes`.
- **`Examples` needs a real `>>>` example.** `test_class_has_doctest_example` fails without
  one, and `test_doctest_examples` runs it, so it has to work. Use a sktime dataset, and
  mark anything slow or needing a download with `# doctest: +SKIP`.
- Cite the paper in `References` as `.. [1] ...`, and refer to it with `[1]_`. Links go there
  too, not inline.
- **Add the API reference entry** where the neighbour's is (step 3). No test catches a
  missing entry, so it is easy to forget, and maintainers end up adding it for you.

## 7. Check the numbers, not only the interface

`check_estimator` proves the estimator follows the API. It doesn't prove it computes the
right thing. A merged metric once needed a maintainer to fix its sign after every
interface test passed. So add at least one test with hand-computed or reference values
in the module's `tests/` folder. When a reference implementation exists, generate expected
values from it with the `sktime-golden-tests` skill. Gate tests that need a soft dependency
with `run_test_for_class`, as the dependency guide shows.

## 8. Verify

Run `check_estimator`, the estimator's test file and pre-commit, using the commands in
"Running tests" in `AGENTS.md`, and report the real result. A skipped test isn't a passed one.

If a test genuinely can't pass, ask the user before skipping it. `tests:skip_by_name` needs
a comment next to it explaining why, ideally linking an issue. Avoid `tests:skip_all`,
which disables every test for the estimator.

## 9. Hand over

Tell the user what you added, the verification output, and anything not verified. Then
suggest running `sktime-pr-check` before opening the PR. New contributors also need to add
themselves to `.all-contributorsrc`, which that skill covers.

## Review findings by scitype

Recurring review comments that the templates don't spell out.

**Forecasters**

- **Don't read `self._y` or `self._X` in `_predict`.** The `remember_data` default is moving
  to `False`, and then they aren't stored. If predict needs history, keep your own copy in
  `_fit`, and extend it in `_update` with `update_data` from `sktime.datatypes`. A forecaster
  that stores context needs an `_update`, or `update` moves the cutoff while `predict` uses
  stale data.
- **`fh` can be anything the user passes,** including non-contiguous horizons such as
  `[2, 5]` and in-sample steps. Convert with `fh.to_relative(self.cutoff)`, compute up to the
  max step, select the requested steps, and index the result with
  `fh.to_absolute(self.cutoff)`.
- **Return the same mtype you received,** with the original variable names and an index
  matching `fh`.
- **Quantiles need `_predict_proba` too.** The default `_predict_proba` assumes a normal
  distribution, so a method that emits quantiles should provide its own, and the two must
  agree.
- Set `capability:multivariate` and `capability:exogenous` only for native support. With
  `False`, sktime vectorizes over variables, and inner methods see `X=None`.

**Classifiers, regressors and clusterers**

- Implement `_predict_proba` when the method has real probabilities. The default gives hard
  0/1 values.
- With a `random_state` parameter, derive every random draw from it. `test_fit_idempotent`
  fits twice and compares, so unseeded randomness makes it flaky.
- Deep learning estimators keep the architecture in `sktime/networks/`, use tiny
  `get_test_params` (one epoch, few filters), and pass custom parameters through to the
  network, which `test_dl_constructor_initializes_deeply` checks.

**Transformers**

- Don't mutate the input `X`. Work on a copy.
- If nothing is learned, set `fit_is_empty` and leave `_fit` out.
- Keep column names and index through the transform where that makes sense, since
  pipelines depend on them.
- If `_inverse_transform` exists, test that it recovers the input.

**Detectors**

- `predict` returns locations in the format the `task` tag requires, not a mask or raw
  scores. The docstring of `_predict` in the template gives the format, and getting it wrong
  is the most common failure in `test_all_detectors.py`.
- Reuse `sktime/detection/costs`, `_penalties.py`, `_change_scores` and `_anomaly_scores`
  rather than re-implementing windowing or penalties.

**Metrics**

- **Get the sign and direction right, and test the value.** Before writing the formula, state
  the definition from the literature, then test a small example worked out by hand.
- Name for the direction: `*Score` is maximized, `*Error` or `*Loss` is minimized. A signed
  quantity isn't an error unless it's non-negative.
- Put the formula in the docstring with `:math:`, in an `r"""` docstring, and make the prose
  and the code agree on what a positive value means.
- A doctest printing a numpy scalar is version-sensitive (`np.float64(0.05)`). Wrap the
  value in `float(...)` rather than skipping the doctest.
- Copy how `multioutput` is handled from an existing metric rather than working it out.

**Parameter estimators**

- The fitted attributes are the output, so document each one under `Attributes`.
