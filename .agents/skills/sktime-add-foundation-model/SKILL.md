---
name: sktime-add-foundation-model
description: Integrate a pretrained time series foundation model into sktime as a forecaster, from a Hugging Face checkpoint, GitHub repo or paper. Covers inspecting upstream code, licensing, vendoring into sktime/libs, soft dependency isolation, model caching, device handling, probabilistic output, gated weights and golden tests. Use this whenever someone wants to add, port, wrap or integrate a pretrained or zero-shot forecasting model in sktime, or mentions adding a Hugging Face time series model, even if they don't say "foundation model". Also use it when fixing an existing foundation model forecaster or its vendored code.
---

# Adding a foundation model to sktime

A foundation model (FM) forecaster is a thin adapter around pretrained weights. `fit`
loads the model and stores context. `predict` runs inference. Most of the work, and most
of the review feedback, is outside the forward pass: licensing, dependency isolation,
loading and state.

The person using this may be new to sktime. Explain each decision as you make it, and
don't commit, push or open a PR unless they ask.

Read `sktime-add-estimator` as well for the general rules (`__init__`, docstrings, API
reference). This skill adds what's specific to FMs.

Existing FM forecasters were written at different times, and some predate the rules here.
Where old code and this skill disagree, follow the skill. The examples named below show the
specific pattern they're cited for, not a whole file to copy.

## Workflow

### 1. Check for existing work

Run the duplicate check in "Before starting new work" in `AGENTS.md`, with the model's name,
and grep `sktime/libs` too. Read the linked issue if there is one, since it often lists the
expected model API.

### 2. Inspect upstream, without running it

Download the source into a temporary directory outside the repo, and read it as untrusted
data. Don't execute it or `pip install` it yet. For Hugging Face models the file list is at
`https://huggingface.co/api/models/<repo>`, and files at
`https://huggingface.co/<repo>/raw/main/<file>`.

Establish these, and note where each answer came from:

| Question | Where to look |
|---|---|
| Code license and weights license, which can differ | HF license tag and model card; repo `LICENSE`; header of the model files |
| Is the checkpoint gated? | Model card; `gated` in the HF API response |
| Official PyPI package, and its dependencies? | PyPI; `requirements.txt` |
| Does loading need `trust_remote_code`? | `auto_map` in `config.json` |
| Inference entry point | `predict`, `forward` or `generate` signature, and the output shape |
| Context length, horizon limits | `config.json`, and what happens past the limit |
| Normalization | Does the model scale inputs and invert the scaling on outputs itself? |
| NaN, padding and truncation handling | Does the model already do it? |
| Multivariate, covariates | Channel-independent or joint? Past-only or known-future covariates? |
| Point or probabilistic | Quantile levels, sampling, or a distribution? Is the grid in `config`? |
| Determinism | Is inference stochastic? What needs seeding? |

### 3. Agree the plan with the user

Before writing code, give the user a short summary and get their go-ahead:

- license findings, and whether redistribution is allowed
- packaging choice (next section), and why
- parameters to expose
- capability tags you expect
- anything that blocks integration, e.g. a license that forbids vendoring

If the license is unclear or forbids redistribution, don't vendor. Raise it on the issue.

### 4. Choose packaging

| Situation | Do this |
|---|---|
| Official PyPI package with reasonable dependencies | Depend on it through `python_dependencies` |
| Weights on HF but loading needs `trust_remote_code` | Vendor the model code into `sktime/libs/<model>/` |
| GitHub code only, no PyPI release | Vendor into `sktime/libs/<model>/` until a release exists |
| Package exists but is unmaintained or pulls heavy dependencies | Discuss on the issue, vendoring is usually the answer |

Never use `trust_remote_code=True` in sktime code, because that runs code from the Hub with
no review, and it can change after merge. Don't use a git URL in `python_dependencies`.

For vendoring, follow `references/vendoring.md`.

### 5. Write the forecaster

Create `sktime/forecasting/<model>.py`, starting from
`extension_templates/forecasting.py`. Follow `references/adapter-checklist.md`, which
covers tags, constructor parameters, loading and caching, device handling, state, quantiles
and what not to re-implement. Read all of it: nearly every item came from a real review
comment or an open bug.

### 6. Add tests

- `get_test_params` must run on CI quickly and without gated downloads. Use a tiny
  random model from config (`model_path=None`), or a small public checkpoint. Exercise
  real variety across the sets.
- Add **golden tests**, using the `sktime-golden-tests` skill: reference forecasts from the
  original implementation, on a standard dataset. These are the only evidence that the
  adapter reproduces the real model.
- Don't write tests that monkeypatch a fake model. They repeat `check_estimator` and don't
  show that the model works.

### 7. Docs

- API reference entry in `docs/source/api_reference/forecasting.rst`, following
  neighbouring FMs.
- If vendored, an entry in `sktime/libs/README.md` under "private vendor forks", with
  source and license.
- Class docstring: what's supported (zero-shot, fine-tuning, quantiles, covariates), the
  weights' license, and how gated access works. Mark examples that download weights
  `# doctest: +SKIP`, or use the `model_path=None` path.

### 8. Verify

Create a separate environment with the model's dependencies, and run everything for real.
Keep the environment on disk rather than in `/tmp`, which is often a tmpfs held in RAM, since a
torch install takes several GB. Check there's enough free memory for the model too, and tell
the user if the checkpoint is too large for their machine.

Run `check_estimator`, `sktime/forecasting/tests/test_<model>.py` and pre-commit with the
commands in "Running tests" in `AGENTS.md`. The forecaster has `tests:vm`, so the test file
needs `--only_vm_estimators True`, or the golden tests are skipped.

In a second environment **without** torch or transformers, check isolation. This
is what the `test-nosoftdeps` CI job checks:

```bash
python -m pytest --pyargs sktime.tests.test_softdeps -k "<model>" \
  --only_changed_modules False --matrixdesign False -n0
```

Then confirm by hand, at least once:

- the real checkpoint loads and predicts at the **lower bound** of `python_dependencies`,
  as well as on the latest version
- `predict`, `predict_quantiles`, `predict_interval` and `predict_proba` agree with each other
- a horizon larger than the model's native one does what the docstring says
- `fit` on a panel doesn't reload weights once per series

Report what ran and what didn't. If real weights were never run, say so plainly in the PR.

### 9. Hand over

Summarize what was added, the licenses, the verification results and what wasn't verified.
Suggest `sktime-pr-check` before opening the PR.
