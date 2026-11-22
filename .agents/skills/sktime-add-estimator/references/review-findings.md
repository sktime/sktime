# Review findings by scitype

Recurring review comments that the templates don't spell out.

## Forecasters

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

## Classifiers, regressors and clusterers

- Implement `_predict_proba` when the method has real probabilities. The default gives hard
  0/1 values.
- With a `random_state` parameter, derive every random draw from it. `test_fit_idempotent`
  fits twice and compares, so unseeded randomness makes it flaky.
- Deep learning estimators keep the architecture in `sktime/networks/`, use tiny
  `get_test_params` (one epoch, few filters), and pass custom parameters through to the
  network, which `test_dl_constructor_initializes_deeply` checks.

## Transformers

- Don't mutate the input `X`. Work on a copy.
- If nothing is learned, set `fit_is_empty` and leave `_fit` out.
- Keep column names and index through the transform where that makes sense, since
  pipelines depend on them.
- If `_inverse_transform` exists, test that it recovers the input.

## Detectors

- `predict` returns locations in the format the `task` tag requires, not a mask or raw
  scores. The docstring of `_predict` in the template gives the format, and getting it wrong
  is the most common failure in `test_all_detectors.py`.
- Reuse `sktime/detection/costs`, `_penalties.py`, `_change_scores` and `_anomaly_scores`
  rather than re-implementing windowing or penalties.

## Metrics

- **Get the sign and direction right, and test the value.** Before writing the formula, state
  the definition from the literature, then test a small example worked out by hand.
- Name for the direction: `*Score` is maximized, `*Error` or `*Loss` is minimized. A signed
  quantity isn't an error unless it's non-negative.
- Put the formula in the docstring with `:math:`, in an `r"""` docstring, and make the prose
  and the code agree on what a positive value means.
- A doctest printing a numpy scalar is version-sensitive (`np.float64(0.05)`). Wrap the
  value in `float(...)` rather than skipping the doctest.
- Copy how `multioutput` is handled from an existing metric rather than working it out.

## Parameter estimators

- The fitted attributes are the output, so document each one under `Attributes`.
