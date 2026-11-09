# sktime PR rubric

The items a reviewer checks. `.github/PULL_REQUEST_TEMPLATE.md` is the source for the PR
items. The rest come from recurring review comments.

## Scope

- **No duplicate.** No other open PR does this, and a linked issue isn't assigned to someone
  else.
- **One concern.** Changes unrelated to the stated purpose go in a separate PR. That includes
  formatting churn in untouched code.
- **No stray files.** No local scripts, scratch notebooks, lockfiles, editor settings or
  generated output in the diff.

## Title and description

- **Title prefix.** Starts with one of: `[ENH]` adding or improving code, `[BUG]` a bugfix,
  `[MNT]` CI or test framework, `[DOC]` documentation or docstrings.
- **Template used.** Sections are present and filled in, not deleted.
- **Issue linked.** `Fixes #1234` where the PR resolves an issue, so it closes on merge.
- **New dependency stated**, if `pyproject.toml` changed.
- **AI disclosure.** The PR template has a hidden comment asking LLMs to begin with the
  header `LLM generated content, by (your model name)`. If the PR description or an issue
  was written by an LLM, including one you draft here, that header goes first. Hidden
  comments are easy for contributors to miss, so point it out when it's absent from text that
  looks generated. A human contributor who wrote the text themselves doesn't need it.

## Contributors file

- **Added to `.all-contributorsrc`.** This is the root file, not `CONTRIBUTORS.md`, which is
  generated. Check with `grep -n '"login": "<handle>"' .all-contributorsrc`.
- **Badges match the work.** `code` for bug fixes or logic, `doc` for documentation,
  `bug` for reporting or diagnosing a bug (plus `code` if it was fixed in the PR),
  `maintenance` for CI and the test framework.
- **Valid JSON.** Check with `python -m json.tool .all-contributorsrc > /dev/null`.

## New estimators

- **API reference entry** in `docs/source/api_reference/<task>.rst`, in the right section. No
  test catches a missing entry, so check the diff for it.
- **`Examples` section** with a working `>>>` doctest. `test_class_has_doctest_example`
  enforces its presence.
- **`get_test_params`** returns at least two different sets, and runs in seconds.
- **`authors` and `maintainers`** are real GitHub handles, and include upstream authors for
  interfaces. `__author__` matches.
- **Soft dependencies.** `python_dependencies` is set, imports happen in methods or
  `__post_init__`, and `"tests:vm": True` is set.
- **`tests:specific`** names the estimator's own test module if it has one.
- **Capability tags** match what the code does.
- **Exported** from `__init__.py` and `__all__` where neighbouring classes are.
- **No leftover** template `todo` comments or template text.

## Docstrings

Compare the docstring to the signature line by line. This is the most common review finding.

- **Parameters complete.** Every `__init__` parameter, including new ones, is documented.
- **`name : type` format,** with spaces around the colon, and `default=` matching the code.
- **Attributes documented** for public fitted attributes.
- **Links under `References`,** cited with `[1]_`, rather than inline URLs in descriptions.
- **Accurate.** The prose matches the code: units, sign, shapes.

## Tests

- **New behaviour is tested,** beyond `check_estimator`, with assertions on values.
- **A bug fix has a test that fails without the fix.** Prove it where possible, in a separate
  worktree so the checkout stays untouched:

  ```bash
  BASE="$(mktemp -d)/base"
  git worktree add "$BASE" FETCH_HEAD
  cp <test file> "$BASE/<test file>"
  (cd "$BASE" && python -m pytest <test file> --only_changed_modules False --matrixdesign False -n0)
  git worktree remove --force "$BASE"
  ```

  `python -m pytest` matters here, because it imports `sktime` from the worktree rather
  than the editable install.
- **Tests are gated.** Tests needing soft dependencies use `run_test_for_class` or
  `_check_soft_dependencies`, with imports inside the test.
- **No fake models.** Tests that monkeypatch the model only prove the mock works.
- **Skips are justified.** `tests:skip_by_name` has a comment explaining why, ideally with an
  issue. `tests:skip_all` needs a strong reason.

**Running tests.** `setup.cfg` turns on change detection, and `tests:vm` estimators are skipped
without an explicit flag. So use:

```bash
pytest <test file> --only_changed_modules False --matrixdesign False -n0 -rs
# for estimators with tests:vm: True, add
--only_vm_estimators True
```

If the summary shows nothing passed, or everything skipped, the item is Not verified, not OK.

`check_estimator` is the direct route for one class:

```python
from sktime.utils.estimator_checks import check_estimator
results = check_estimator(MyEstimator)
print({k: v for k, v in results.items() if v != "PASSED"})
```

## Dependencies

- A new soft dependency used by a single estimator lives in the `python_dependencies` tag only,
  not in the `pyproject.toml` extras.
- New core dependencies need agreement from core developers.
- Version bounds are justified. Lower bounds were actually tried.

## Compatibility

- Changing a public name, signature, default or algorithm needs a deprecation. See the
  `sktime-deprecation` skill for the policy.
- A deprecation has a warning, a docstring note, `todo <version>` markers and tests.

## Code

- No checks that repeat the base class, no dead branches, no defensive shape conditionals where
  inner mtypes guarantee the input.
- No debug prints or commented-out code.
- Comments explain non-obvious reasons.
- Standard private signatures are kept, even if CodeQL flags a mismatch.

## CI

- Pre-commit passes. `pre-commit run --files <changed files>`.
- For a PR: `gh pr checks`. For each failure, say whether it's caused by this change or
  unrelated. Check `main` before calling it unrelated.
- For a local branch: CI hasn't run, so the item is Not verified.
