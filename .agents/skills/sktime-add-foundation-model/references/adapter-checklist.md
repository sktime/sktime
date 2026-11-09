# Foundation model forecaster checklist

These rules come from maintainer reviews and open bugs. Issue numbers point to the bug
behind a rule. Go through every section, and mark what doesn't apply rather than skipping
it.

## Contents

- [Tags](#tags)
- [Constructor parameters](#constructor-parameters)
- [Loading and caching](#loading-and-caching)
- [Device handling](#device-handling)
- [State and update](#state-and-update)
- [Horizon and output](#horizon-and-output)
- [Probabilistic output](#probabilistic-output)
- [Covariates](#covariates)
- [Don't duplicate the base class or the model](#dont-duplicate-the-base-class-or-the-model)
- [Code quality](#code-quality)
- [Tests](#tests)

## Tags

```python
_tags = {
    "category": ["deep_learning", "foundation_model"],
    "capability:exogenous": False,
    "capability:insample": False,
    "capability:pred_int": True,
    "requires-fh-in-fit": False,
    "authors": ["upstream-handle", "your-handle"],
    # upstream-handle for <org>/<repo>
    "maintainers": ["your-handle"],
    "python_dependencies": ["transformers[torch]>=4.57,<5"],
    # CI and test flags
    # -----------------
    "tests:vm": True,
    "tests:libs": ["sktime.libs.<model>"],  # if vendored
    "tests:specific": ["sktime.forecasting.tests.test_<model>"],
}
```

The general rules for `authors`, `maintainers` and the soft dependency tags are in
`sktime-add-estimator` step 4 and "Soft dependencies" in `AGENTS.md`. For FMs also:

- Check capability tags against real behaviour. Test them, don't assume.
- Verify the upstream authors' GitHub handles, e.g. against upstream's commit authors.
- `tests:skip_all` is not how to handle gated weights, since it disables everything. See
  [Tests](#tests).

## Constructor parameters

- Question each parameter. Remove ones with no real use, or that can be set some other way,
  such as an `ignore_deps` flag "for testing".
- **`license_accepted`** depends on the weights' license. If it's non-commercial, proprietary
  or otherwise different from sktime's, add `license_accepted=False`, which must be set to
  use the model (see `sktime/forecasting/t0.py` and `falcon_x.py`). `get_test_params` and
  golden tests then set it. If the license doesn't call for it, leave it out.
- For pass-through dicts (`model_kwargs`, `forecast_kwargs`), link in the docstring to where
  options are documented. Put the link under `References`, and cite it with `[1]_`.
- Add `deterministic=False` if inference samples. With `True`, seed right before the call, as
  in `transformers.set_seed(42)`.
- **Fixing an upstream bug:** agree it with the user first. The pattern accepted for
  `TEMPOForecaster` (#11073): fix it in the vendored code, behind one boolean `__init__`
  parameter (`upstream_compatible=True`) that defaults to upstream's behaviour. Its
  docstring says what each value does and which upstream bug it's about. Put the parameter
  in the `_multiton` key if it changes loading, use the non-default value in one
  `get_test_params` set, and list the change in the vendored README.

## Loading and caching

- **Cache with `_multiton`** from `sktime.utils.singleton`, keyed on all loading arguments.
  Otherwise, each `fit` reloads weights, and with `y_inner_mtype` as `pd.DataFrame`,
  vectorization reloads them once per series. Don't use `lru_cache` or a custom cache.
  Examples: `_CachedFalconTST` in `falcon_tst.py`, `chronos2.py`.
- **The key has to contain everything that affects loading**: path, config, device, dtype,
  quantization. Build it deterministically: `str(sorted(key.items()))`.
- **The cached model is shared across forecasters, so don't mutate it.** No training on
  it, no per-instance `.to()`, no config edits at predict time. If you add fine-tuning,
  copy it first (#11579).
- **Call `.eval()` on every loading path**, including the path that builds a model from
  config. Otherwise dropout stays on during predict (#11310).
- **No `trust_remote_code`.**
- **Gated checkpoints:** catch `GatedRepoError` when loading, and re-raise it with a
  message telling the user how to get access.
- **Loading checks must catch the real failure.** A missing-keys check that passes while
  weights silently fail to load gives false assurance. Check what actually goes wrong, or
  leave it out.
- **Compare the checkpoint with the config.** With `load_state_dict(strict=False)`, a config
  that asks for more layers than the checkpoint holds leaves the extra layers random, so
  forecasts change between processes. `TEMPO` (#11073) had `gpt_layers=6` and 3-layer
  checkpoints. List the checkpoint's keys and the model's, and raise any mismatch with the
  user.
- Don't add an argument that has only one possible value, e.g. `model_type` when upstream has
  just one class.

## Device handling

- Take `device` or `device_map`, defaulting to `"cpu"`. If you offer `"auto"`, resolve it into a
  private `self._device`, and leave `self.device` as passed (`_resolve_device` in
  `kronos.py`). The resolved device has to be what goes in the cache key.
- Move inputs to the model's device with `.to(model.device)` and `.to(model.dtype)`. Don't
  hardcode `"cuda"`.
- Convert outputs with `.detach().float().cpu().numpy()`.
- Wrap inference in `torch.no_grad()`, or use the upstream entry point that does.
- Fix `device="cpu"` in `get_test_params` and golden tests. CI has no GPU, and MPS is
  disabled on macOS runners (`sktime/tests/_test_vm.py`).
- Check `test_persistence_via_pickle` and `deepcopy` work for a fitted model. If the upstream
  object can't be pickled, drop it in `__getstate__` and reload lazily, as `timesfm3.py`
  does. A fitted model mustn't carry device-bound tensors that fail on a different
  machine (#10599).

## State and update

The general rules are in "Common review findings" in
`../../sktime-add-estimator/references/forecasting.md`: don't read `self._y` or `self._X`,
keep your own context (e.g. `self._cur_y`, `self._cur_X`), and implement `_update` with
`update_data`. Then set `"capability:update": True`. For FMs also:

- In `_update`, raise an error if `X` was passed to `_fit` but not `_update`, or the other
  way round. Otherwise `_cur_X` drifts out of line with `_cur_y`.

## Horizon and output

Bugs here produce plausible numbers, so check each with real output (#11347 was all three):

- **Scale:** forecasts are on the scale of `y`. Check whether the model normalizes
  internally, and invert it exactly once.
- **Non-contiguous `fh`:** `[2, 5]` has to return steps 2 and 5, as `forecasting.md`
  describes.
- **Long horizons:** decide what happens past the model's native horizon, and document it.

## Probabilistic output

- Implement `_predict_proba` as well as `_predict_quantiles` (see `forecasting.md`), e.g.
  with `HistogramQPD` from `skpro.distributions`, with `tails="mass"`, as in `timesfm3.py`.
- `predict`, `predict_quantiles` and `predict_proba` must agree (#10566, #10533). Add a test.
- Use the model's own level list (`model.quantile_levels`, `config`) over a hardcoded grid
  or index, including the median index for the point forecast.
- **Clamp `alpha` outside the native grid** to the outer quantiles rather than raising. This is
  consistent with `tails="mass"`, and otherwise `predict_interval(coverage=0.99)` raises
  on a 0.01-0.99 grid. Some older forecasters raise here. Don't copy them.
- Interpolate between levels with `np.interp` directly. It returns exact values at grid points,
  so you don't need a nearest-level helper.
- If you forward arbitrary `alpha` values upstream, confirm upstream really handles levels off
  its native grid, and say in the docstring how.

## Covariates

- Check that dtypes survive joining history and future `X`. `pd.concat` of two categoricals
  with different categories falls back to the values' dtype, so numeric labels become `int64`,
  and upstream treats them as numeric. Convert to object first, and test with different
  categories on each side.
- Check models against datasets with exogenous columns and multiple variables, such as
  Longley (#11535).

## Don't duplicate the base class or the model

- Remove checks the base class enforces: positive `fh` with `capability:insample=False`, `fh`
  conversion, vectorization.
- Remove truncation, padding, horizon capping or NaN handling that upstream already does,
  especially with hardcoded constants that can drift from `config`. If upstream takes the
  limit as an option (`max_context` in `from_pretrained`), pass it there.
- Where non-obvious preprocessing remains, add a comment saying why.
- Don't re-validate what Python catches. `isinstance(kwargs, dict)` checks and reserved-key
  blocks go, since passing a key twice raises `TypeError`. Handle only keys that would
  otherwise be silently overwritten.

## Code quality

- No changes unrelated to the model in the PR.
- Keep standard private signatures, e.g. `_predict_quantiles(self, fh, X, alpha)`, even if
  CodeQL flags them.
- Name the shared "run the model once" helper for what it does, like `_run_forecast`.
- Inline helpers that are used once and only a couple of lines long.
- Remove wrappers that only forward to another method, and helpers that duplicate each other.
- Remove dead branches: DataFrame handling when `y_inner_mtype` is `pd.Series`, or `if arr.ndim`
  checks, when inner mtypes already guarantee the shape.
- Remove code that restates Python defaults, e.g. a `__setstate__` doing only
  `self.__dict__.update(state)`.
- Avoid repeated `deepcopy` of already-copied options. A plain `dict(...)` is enough.
- Use `pd.api.types`, like `is_numeric_dtype`, rather than comparing `dtype.name` to strings.

## Tests

- `get_test_params`: at least two sets, on `device="cpu"`, exercising options from
  `model_kwargs` or `forecast_kwargs`, not only different devices.
- **Gated weights:** with `model_path=None`, initialize from the default config without
  downloading anything (see `timer_s1.py`, `falcon_tst.py`). Tests can then run with no
  authentication. Warn that the weights are random. That replaces `tests:skip_all`.
- Golden tests with real weights, or from a seeded untrained model when the checkpoint is gated.
- No monkeypatched fake models, and no tests of trivial behaviour, such as a license check
  raising.
