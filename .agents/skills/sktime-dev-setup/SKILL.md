---
name: sktime-dev-setup
description: Set up, check or repair a local sktime development environment. Creates a virtual environment, makes an editable install with the dev, test and docs dependencies, installs pre-commit, wires up the upstream remote, and builds the docs locally. Use this whenever someone wants to start contributing to sktime, set up or recreate their environment, build the docs, get pre-commit working, or hits import, install or "module not found" errors in a fresh clone, even if they only say "get me ready to work on sktime".
---

# sktime dev setup

The source is "Full developer setup" in `docs/source/installation.rst`, plus
`docs/source/developer_guide/git_workflow.rst` for remotes and "Documentation Build" in
`docs/source/developer_guide/documentation.rst`. The guide uses conda. This skill uses a
plain venv and `uv`, which most contributors now use, and adds the checks the guide leaves
out.

Run every step from the repo root. Check what exists before changing anything: if a venv,
remote or hook is already there and works, keep it and say so.

## 1. Check the clone and remotes

```bash
git rev-parse --show-toplevel
git remote -v
```

Contributors push to their fork and open PRs against `sktime/sktime`. One remote must point
to `github.com/sktime/sktime`, for fetching `main`. Names vary (`upstream`, or `origin` if
the clone came from upstream and the fork was added as a second remote), so read them rather
than assuming. If none points to sktime, add it:

```bash
git remote add upstream https://github.com/sktime/sktime.git
```

Don't rename or remove existing remotes without asking.

## 2. Create the virtual environment

`requires-python` in `pyproject.toml` gives the supported range. Pick a version in the
middle of it, not the newest: many soft dependencies (torch, tensorflow, statsforecast, ...)
publish wheels late. CI's main jobs use the lowest supported version, and Read the Docs uses
the one in `.readthedocs.yml`.

```bash
grep requires-python pyproject.toml
uv venv .venv --python 3.12
source .venv/bin/activate
```

`.venv` is in `.gitignore`. If the user prefers conda or an existing env, use that, and
follow the guide's steps instead.

## 3. Editable install

```bash
uv pip install -e ".[dev,docs]"
```

- `dev` brings pytest, its plugins and pre-commit. `tests` holds pinned versions of the
  same, used by CI; `dev` is enough locally.
- `docs` is only needed to build the docs. Leave it out if the user doesn't need them.
- **Soft dependencies:** don't install `all_extras` by default. It's large, slow, and often
  fails to resolve on newer Python versions. Install what the work needs, e.g. the
  estimator's `python_dependencies` tag, or a scitype extra such as `forecasting`.
- Use `uv pip install`, not `uv sync` or `uv add`. Those write a `uv.lock` and may edit
  `pyproject.toml`, and sktime tracks neither change. If a `uv.lock` appears, don't commit it.

Check the install points at the clone, not at site-packages:

```bash
python -c "import sktime, sys; print(sktime.__version__, sktime.__file__, sys.version)"
```

The path must be inside the repo. If it isn't, another sktime is shadowing it: uninstall it
and reinstall the editable one.

## 4. pre-commit

```bash
pre-commit install
pre-commit run --files sktime/__init__.py
```

The config is `.pre-commit-config.yaml` (ruff format, ruff check, file hygiene, shellcheck).
The first run downloads the hook environments, so it takes a while. Use
`pre-commit run --files <changed files>` on real changes. Don't run `--all-files`: it
reformats files untouched by the PR, and the diff is then rejected as unrelated.

## 5. Smoke test

Run a small test with the flags in "Running tests" below, so change detection doesn't skip
it:

```bash
python -m pytest sktime/forecasting/tests/test_naive.py --only_changed_modules False \
  --matrixdesign False -n0 -q
```

or run `check_estimator` on a simple estimator such as `NaiveForecaster`. Report the
summary line. Everything skipped means the setup isn't tested yet.

## 6. Docs (only if asked)

Install `docs` (step 3), then:

```bash
cd docs/source
make html            # output in docs/build/html, which is git-ignored
make clean html      # full rebuild
```

- The first build takes up to about 20 minutes, since it imports every estimator for the
  API reference. Run it in the background, and tell the user.
- Notebooks aren't executed (`nbsphinx_execute = "never"` in `conf.py`), but nbsphinx needs
  the `pandoc` binary, which pip doesn't provide. Check with `pandoc --version`, and ask the
  user to install it from their system package manager if it's missing.
- `docs/source/examples` links to the top-level `examples/`. If it's empty or broken, run
  `./symlink_examples.sh` from `docs/source`.
- For live reloading: `uv pip install sphinx-autobuild`, then `make autobuild`.
- Warnings don't fail the build (`fail_on_warning` is off in `.readthedocs.yml`), so check
  the log for warnings on the pages that changed, e.g. a new API reference entry.
- The published site is built from a separate repo, `sktime/sktime-webpage-docs`. A local
  build only checks that the pages render.

## Running tests

The other sktime skills point here.

`setup.cfg` turns on change detection and matrix sub-sampling (the options are described in
the root `conftest.py`), and `run_test_for_class` skips `tests:vm` estimators in a normal run.
So a plain `pytest` often runs nothing. Use:

```bash
python -m pytest <test file> --only_changed_modules False --matrixdesign False -n0 -rs
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

## 7. Report

Tell the user:

- the Python version, venv path and how to activate it
- which extras were installed, and any soft dependencies left out
- the remotes, and which one is upstream
- whether pre-commit is installed, and the smoke test summary line
- the docs build result and its warning count, if built
- anything that failed or was skipped, with the error

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| `ModuleNotFoundError` for a third-party package | A soft dependency isn't installed. Install the one the estimator needs, not `all_extras` |
| `import sktime` loads from site-packages | Another sktime install, or the wrong venv is active |
| Tests show "0 selected" or all skipped | Missing flags, see "Running tests" above |
| Resolver fails on install | Python version too new for a dependency; recreate the venv on an older one |
| Docs build fails on notebooks | `pandoc` missing |
| A commit is blocked by pre-commit | The hooks fixed files; review, `git add` and commit again |
