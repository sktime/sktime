---
name: sktime-add-estimator
description: Add a new estimator to sktime, end to end, for contributors, or write an sktime compatible estimator in a third party extension package or a local, closed codebase. Covers choosing the scitype and extension template, tags, get_test_params, docstrings, the API reference entry, and check_estimator. Use this whenever someone wants to add, implement, port, wrap or interface a forecaster, transformer, classifier, regressor, clusterer, detector, metric, splitter, aligner, distance, kernel or parameter estimator in sktime, even if they only name the algorithm or paper and never say "estimator". For pretrained time series foundation models, use sktime-add-foundation-model instead.
---

# Adding an estimator to sktime

The person using this is often a first-time sktime contributor. Do the work, but
explain the sktime-specific decisions as you make them, so they can defend the PR in
review. Don't commit, push or open a PR unless they ask.

**If the estimator lives outside sktime,** in a third party extension package or a local,
closed codebase, see [Outside sktime](#outside-sktime) first. Most of this skill still
applies, but the steps specific to sktime's repo don't.

The sources for this work are in the repo. Read them rather than working from memory:

- `docs/source/developer_guide/add_estimators.rst`: choosing the scitype, how to fill in a
  template, `check_estimator`, and what adding to sktime itself requires.
- `extension_templates/`: one template per scitype. Each one's header and `todo` comments
  give the base class, the mandatory and optional methods, the tags and the input guarantees
  for the inner methods. They're authoritative for the method contracts.

This skill adds what those leave out: the decisions reviewers push back on, and how to check
the result.

## 1. Check it should be added

1. Check for duplicates, the most common reason a PR gets closed:

   ```bash
   grep -rli "<name>" sktime docs/source/api_reference
   gh pr list --repo sktime/sktime --state open --search "<name>"
   gh issue list --repo sktime/sktime --state open --search "<name>"
   ```

   If an open PR covers it, or the issue is assigned to someone else, stop and tell the
   user. They should coordinate there instead of opening a competing PR.
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

Then read the scitype's section in `references/review-findings.md`.

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

**If the estimator has a soft dependency,** follow "Isolating soft dependencies to
estimators" in `docs/source/developer_guide/dependencies.rst`: imports inside methods or
`__post_init__`, the `python_dependencies` tag, `"tests:vm": True`, and tests gated with
`run_test_for_class`. An import at module top level or in `__init__` makes `import sktime`
fail for users without the package. The guide also covers the two utilities in
`sktime.utils.dependencies`:

- **`_safe_import("pkg.mod.Name")`** for a name that has to exist when the module is
  imported, such as a base class (`nn.Module`) or a decorator. It returns a mock if the
  package is missing. Not as the parent of a `dataclass`. Prefer a method level import
  where one works.
- **`_check_soft_dependencies`** outside an estimator, e.g. in a utility function or to gate
  a test or a code block. Inside an estimator the `python_dependencies` tag does this.

The guide leaves out three things:

- **`"tests:specific": ["sktime.<path>.tests.test_<name>"]`** names the estimator's own test
  module. Without it the VM job logs "No object specific tests found", never runs the file,
  and stays green.
- **`"tests:libs": ["sktime.libs.<name>"]`** does the same for vendored code in
  `sktime/libs`, so changes to it trigger the estimator's tests.
- A dependency used by one estimator lives in its `python_dependencies` tag only, not in the
  `pyproject.toml` extras. The base class raises the missing-dependency error from the tag,
  so don't add your own try/except or warning.

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
"Running tests" in `../sktime-dev-setup/SKILL.md`, and report the real result. A skipped
test isn't a passed one.

If a test genuinely can't pass, ask the user before skipping it. `tests:skip_by_name` needs
a comment next to it explaining why, ideally linking an issue. Avoid `tests:skip_all`,
which disables every test for the estimator.

## 9. Hand over

Tell the user what you added, the verification output, and anything not verified. Then
suggest running `sktime-pr-check` before opening the PR. New contributors also need to add
themselves to `.all-contributorsrc`, which that skill covers.

## Outside sktime

For an estimator in a third party extension package (open or closed) or a local codebase,
follow the steps above with these changes:

- **Skip** the duplicate check, the API reference entry, `.all-contributorsrc` and
  `sktime-pr-check`. These apply only to PRs to sktime. "Adding an sktime compatible
  estimator to sktime" in the guide lists what sktime itself adds on top.
- **Placement** follows the user's package, not an sktime neighbour. Still copy the
  extension template, and still inherit from the sktime base class.
- **Soft dependencies:** the `python_dependencies` tag still works, but whether a package is
  a hard dependency of the user's own package is their call. The `tests:*` CI tags have no
  effect outside sktime's CI.
- **Testing:** use "Testing within a third party extension package" in the guide:
  `check_estimator`, or `parametrize_with_checks` in a pytest suite. The pytest flags in
  `sktime-dev-setup` are for sktime's repo only.
- **`authors` and `maintainers`** may hold the user's own handles or names, since the
  GitHub handle rule is for sktime's credit system.
