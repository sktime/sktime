# AGENTS.md - Guidance for AI Coding Agents

This file provides pointers for AI coding agents (Copilot, Cursor, OpenHands,
etc.) working on the `sktime` codebase.

## Pre-commit Hooks

sktime uses [pre-commit](https://pre-commit.com/) for automated code quality
checks. The configuration is in `.pre-commit-config.yaml` at the repo root.

### Setup and usage

```bash
pip install pre-commit
pre-commit install          # install hooks into your local git clone
pre-commit run --all-files  # run all hooks on the entire codebase
pre-commit run <hook-id>    # run a specific hook, e.g. ruff-format
```

### Hooks configured

| Hook repo | Hooks |
|-----------|-------|
| `pre-commit-hooks` | check-added-large-files, check-case-conflict, check-merge-conflict, check-symlinks, check-yaml, debug-statements, end-of-file-fixer, requirements-txt-fixer, trailing-whitespace, mixed-line-ending |
| `ruff-pre-commit` | **ruff-format** (formatter), **ruff-check --fix** (linter) |
| `check-manifest` | check-manifest (manual stage only) |
| `shellcheck-py` | shellcheck |

Always run `pre-commit run --all-files` before committing to ensure compliance.

## Code Formatting

sktime uses **Ruff** for both formatting and linting (via pre-commit):

- **Formatter:** `ruff format` — formats code in place.
- **Linter:** `ruff check --fix` — applies auto-fixable lint rules.

You can also run these directly:

```bash
ruff format .
ruff check --fix .
```

Additional style rules:
- Lines end with LF (not CRLF).
- No trailing whitespace.
- Files end with a single newline.

## Adding a New Estimator

### 1. Use the extension templates

The `extension_templates/` directory contains boilerplate templates for new
estimators. Pick the template matching your estimator type:

| Template file | Estimator type |
|---------------|---------------|
| `classification.py` | Time series classifier |
| `clustering.py` | Time series clusterer |
| `detection.py` | Time series detector (anomaly/changepoint) |
| `forecasting.py` | Forecaster (full interface) |
| `forecasting_simple.py` | Forecaster (simplified) |
| `forecasting_supersimple.py` | Forecaster (minimal) |
| `forecasting_global_supersimple.py` | Global forecaster (minimal) |
| `transformer.py` | Transformer (full interface) |
| `transformer_simple.py` | Transformer (simplified) |
| `transformer_supersimple.py` | Transformer (minimal) |
| `transformer_supersimple_features.py` | Feature-extracting transformer (minimal) |
| `early_classification.py` | Early time series classifier |
| `param_est.py` | Parameter estimator |
| `split.py` | Splitter |
| `alignment.py` | Sequence aligner |
| `dist_kern_tab.py` | Tabular distance/kernel |
| `dist_kern_panel.py` | Panel distance/kernel |
| `metric_detection.py` | Detection metric |
| `catalogue.py` | Catalogue |

Copy the relevant template into the appropriate `sktime/<module>/` subpackage,
rename it, and implement the required methods as documented in the template.

### 2. Register in the module's `__init__.py`

Add your estimator class to the `__all__` list in the subpackage's
`__init__.py` so it is importable from the public API.

### 3. Add to the API reference docs

Add your estimator to the matching `.rst` file in
`docs/source/api_reference/`. Follow the existing pattern:

```rst
.. currentmodule:: sktime.<module>.<submodule>

.. autosummary::
    :toctree: auto_generated/
    :template: class.rst

    YourNewEstimator
```

Look at existing entries in the same file for the correct `currentmodule` path
and section placement.

## Contributing Guide

The full contributing guide is at:
<https://www.sktime.net/en/latest/get_involved/contributing.html>

See also `CONTRIBUTING.md` in the repo root.
