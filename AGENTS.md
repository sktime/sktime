# sktime agent notes

Commands and rules that apply to every task in this repo. The skills in `.agents/skills/`
point here instead of repeating them. Where a developer guide page covers a topic, it is
the source, and this file only adds what it leaves out.

Don't commit, push or open a PR unless the user asks.

## Before starting new work

Duplicates are the most common reason a PR gets closed. Search first:

```bash
grep -rli "<name>" sktime docs/source/api_reference
gh pr list --repo sktime/sktime --state open --search "<name>"
gh issue list --repo sktime/sktime --state open --search "<name>"
```

If an open PR covers it, or the issue is assigned to someone else, stop and tell the user.
They should coordinate there instead of opening a competing PR.

## Running tests

`setup.cfg` turns on change detection and matrix sub-sampling (the options are described in
the root `conftest.py`), and `run_test_for_class` skips `tests:vm` estimators in a normal run.
So a plain `pytest` often runs nothing. Use:

```bash
pytest <test file> --only_changed_modules False --matrixdesign False -n0 -rs
# if the estimator has "tests:vm": True, which is what the CI VM job runs, add
--only_vm_estimators True
```

Read the summary line. "0 selected", or everything skipped, means nothing was tested. The
`-rs` output says why: `False_requires_vm` means the VM flag is missing,
`False_required_deps_missing` means a dependency isn't installed. Report a skipped test as
not run, never as passed.

For one estimator, `check_estimator` is the direct route:

```bash
python -c "
from sktime.utils.estimator_checks import check_estimator
from sktime.<path> import MyEstimator
results = check_estimator(MyEstimator)
failed = {k: v for k, v in results.items() if v != 'PASSED'}
print(len(results), 'run,', len(failed), 'failed')
print(failed)
"
```

Narrow a failure with `tests_to_run="test_name"` or
`fixtures_to_run="test_name[MyEstimator-0]"`, then add `raise_exceptions=True` for the
traceback.

Lint with `pre-commit run --files <changed files>`.

## Soft dependencies

Follow "Isolating soft dependencies to estimators" in
`docs/source/developer_guide/dependencies.rst`: imports inside methods or `__post_init__`,
the `python_dependencies` tag, `"tests:vm": True`, and tests gated with
`run_test_for_class`. The guide leaves out three things:

- **`"tests:specific": ["sktime.<path>.tests.test_<name>"]`** names the estimator's own test
  module. Without it the VM job logs "No object specific tests found", never runs the file,
  and stays green.
- **`"tests:libs": ["sktime.libs.<name>"]`** does the same for vendored code in
  `sktime/libs`, so changes to it trigger the estimator's tests.
- A dependency used by one estimator lives in its `python_dependencies` tag only, not in the
  `pyproject.toml` extras. The base class raises the missing-dependency error from the tag,
  so don't add your own try/except or warning.

## Pull requests

`.github/PULL_REQUEST_TEMPLATE.md` is the source for the title prefix, the description
sections, the `.all-contributorsrc` entry and badges, and the new-estimator checklist. Its
hidden last comment asks LLMs to begin any PR description or issue they write with the header
`LLM generated content, by (your model name)`.
