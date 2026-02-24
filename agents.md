# Agents Guide

This document provides guidance for automated agents and contributors working with the sktime repository. It highlights key practices required to pass code quality checks and to ensure new estimators integrate correctly with the library.

## Pre-commit and Code Quality

sktime uses pre-commit hooks to enforce code quality and formatting.

Before submitting a pull request:

1. Install pre-commit:

   ```bash
   pip install pre-commit
Install the hooks:

pre-commit install
Run checks locally:

pre-commit run --all-files
Pull requests must pass all pre-commit checks.

Adding New Estimators to the Documentation
When introducing a new estimator:

Ensure the estimator appears in the appropriate API reference section.

Update documentation in the docs/ directory as required.

Follow existing estimator documentation patterns.

Failure to register estimators in the API reference may result in incomplete documentation builds.

Estimator Implementation Requirements
New estimators must follow the templates provided in:

extension_templates/
These templates ensure consistency with:

sktime estimator API

testing expectations

documentation structure

tag configuration

Agents and contributors should always start from the relevant template when implementing new estimators.

General Contribution Expectations
Follow the guidelines in CONTRIBUTING.md.

Keep pull requests focused and minimal.

Ensure all tests and pre-commit checks pass.

Maintain consistency with existing code style and documentation tone.

Notes for Automated Agents
Automated agents should:

avoid large refactors unless explicitly requested

limit changes to the scope of the issue

ensure reproducibility of results

respect sktime design conventions

When in doubt, prefer minimal, well-tested changes.


---

# 🧪 Before You Commit (important)

Run:

```bash
pre-commit run --all-files
If it reformats — commit again.
