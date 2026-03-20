# Agents Guide for `sktime`

This document provides guidance for AI coding agents working on `sktime`.
It summarizes key conventions, formatting rules, and project structure
that agents should follow when generating or modifying code.

For full contributor documentation, see the
[contributing guide](https://www.sktime.net/en/latest/get_involved/contributing.html).

## Code Quality and Formatting

`sktime` uses [`pre-commit`](https://pre-commit.com/) hooks to enforce code quality.
All contributions must pass the `pre-commit` checks to pass CI.

### Setting up `pre-commit`

Install `sktime` in development mode and set up `pre-commit`:

```bash
pip install -e ".[dev]"
pre-commit install
```

Once installed, `pre-commit` will automatically run on changed files at each commit.
To run checks manually on all files:

```bash
pre-commit run --all-files
```

### Code formatting rules

The following tools are used for code formatting and linting
(configured in `.pre-commit-config.yaml` and `pyproject.toml`):

- **[ruff](https://docs.astral.sh/ruff/)** for formatting and linting.
  - Line length: 88 characters.
  - Target Python version: 3.10.
  - Format: `ruff format` (run via `pre-commit`).
  - Lint: `ruff check --fix` (run via `pre-commit`).
- **[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html)**
  style docstrings (numpy docstring convention).
- **PEP 8** coding style.

Key `sktime`-specific conventions:

- Use underscores to separate words in non-class names (e.g., `n_instances`).
- Use absolute imports for references inside `sktime`.
- Do not use `import *`.
- Avoid multiple statements on one line.

Full coding standards:
[coding standards](https://www.sktime.net/en/latest/developer_guide/coding_standards.html).

## Adding New Estimators

New estimators should follow the extension templates in the
[`extension_templates`](https://github.com/sktime/sktime/tree/main/extension_templates)
directory. Each template corresponds to an estimator type (scitype):

| Scitype | Template |
|---------|----------|
| Forecaster | `extension_templates/forecasting.py` |
| Classifier | `extension_templates/classification.py` |
| Regressor | `extension_templates/regression.py` |
| Transformer | `extension_templates/transformer.py` |
| Clusterer | `extension_templates/clustering.py` |
| Detector | `extension_templates/detection.py` |
| Parameter Estimator | `extension_templates/param_est.py` |
| Alignment | `extension_templates/alignment.py` |
| Splitter | `extension_templates/split.py` |

Simpler variants are available for some types
(e.g., `forecasting_simple.py`, `forecasting_supersimple.py`).

### Key points for new estimators

- Copy the appropriate extension template to the target module location.
- Implement the private methods (e.g., `_fit`, `_predict`), not the public ones.
- Set estimator tags to declare capabilities and input type expectations.
- Implement `get_test_params` with parameters that cover major internal cases.
- Do not override public methods like `fit` or `predict` —
  implement `_fit` and `_predict` instead.

For full guidance, see the
[implementing estimators](https://www.sktime.net/en/latest/developer_guide/add_estimators.html)
developer guide.

## Documentation

### API reference

When adding a new estimator to `sktime`, it must also be added to the API reference.
The API reference files are located in `docs/source/api_reference/`
and are organized by module:

- `forecasting.rst` — forecasters
- `classification.rst` — classifiers
- `regression.rst` — regressors
- `transformations.rst` — transformers
- `clustering.rst` — clusterers
- `detection.rst` — detectors
- `param_est.rst` — parameter estimators

Follow the existing patterns in these `.rst` files when adding new entries.

### Docstring format

All public classes, methods, and functions must have docstrings
in [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) format.

New estimators should include an `Examples` section in their docstrings
with one or more usage examples.

For full documentation standards, see the
[documentation guide](https://www.sktime.net/en/latest/developer_guide/documentation.html).

## Dependencies

- New estimators should avoid adding core dependencies.
- If a new soft dependency is needed, set the `python_dependencies` tag
  on the estimator and follow the
  [dependencies guide](https://www.sktime.net/en/latest/developer_guide/dependencies.html).

## Testing

Run tests for a specific estimator:

```bash
pytest -k "EstimatorName"
```

Run the full test suite:

```bash
pytest
```

Use `check_estimator` for interface conformance testing:

```python
from sktime.utils.estimator_checks import check_estimator
check_estimator(MyEstimator)
```

## PR Conventions

- PR titles must start with a tag: `[ENH]`, `[BUG]`, `[DOC]`, or `[MNT]`.
  - `[ENH]` — new feature or enhancement
  - `[BUG]` — bug fix
  - `[DOC]` — documentation improvement
  - `[MNT]` — maintenance (CI, tests, packaging)
- Reference related issues using keywords (e.g., `Fixes #1234`).
- Add yourself to `.all-contributorsrc` in the repository root.
