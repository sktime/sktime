# Vendoring model code into `sktime/libs`

Vendoring copies upstream model code into `sktime/libs/<model>/`, as a private fork. It
avoids `trust_remote_code` and removes a dependency on a package that may not exist.

Examples: `sktime/libs/timer_s1/` (from Hugging Face), `sktime/libs/kronos/` and
`sktime/libs/windfm/` (from GitHub).

## License comes first

- Confirm the license allows redistribution. This is a check on the **code** license.
  The weights' license is separate, and affects `license_accepted` instead (see
  `adapter-checklist.md`).
- Use the license of the **source the files came from**. Check the HF repo's license tag and
  model card, or the specific repo. Don't assume the root `LICENSE` of a monorepo covers it.
- The `__init__.py` header, any `LICENSE` file and the class docstring must all agree.
- Keep the **full license text**, not a summary or a link. MIT and Apache both require it.

## Copy only what inference needs

Pin the source to one commit. Copy the configuration, model and generation files you use.
Leave out training loops, demos and unrelated model variants.

If a dependency is only pulled in by upstream code and is easy to remove, such as `einops`
for one `rearrange`, removing it is worthwhile. Don't rewrite the model to do so. Keep the
code close to upstream, because future syncs rely on being able to compare the two.

## The `__init__.py`

The docstring records provenance. Follow `sktime/libs/timer_s1/__init__.py`:

```python
"""Python implementation of <Model>.

Unofficial fork of the ``<org>/<repo>`` model code from huggingface,
available at https://huggingface.co/<org>/<repo>.

sktime migration: <year>, <month>
Version <full commit hash> release: <year>, <month> (<original author>)

Original author: <original author>

<year> releases subject to following license:

All code contained is released under the license below.

<full license text>
"""

from skbase.utils.dependencies import _check_soft_dependencies

if _check_soft_dependencies("torch", "transformers", severity="none"):
    from sktime.libs.<model>.configuration_<model> import <Model>Config
    from sktime.libs.<model>.modeling_<model> import <Model>ForPrediction
else:

    class <Model>Config:
        """Placeholder for <Model>Config."""

    class <Model>ForPrediction:
        """Placeholder for <Model>ForPrediction."""
```

Don't copy or invent a `__version__`. It's misleading for code pinned to a commit, which
the docstring already records.

## Soft dependency isolation

`sktime/libs/<model>` gets imported in the `test-nosoftdeps` CI job, where torch and
transformers are not installed. An unguarded import fails `sktime/tests/test_softdeps.py`.

Use the guard above in `__init__.py`, and `_safe_import` for module-level imports inside
the vendored files:

```python
from sktime.utils.dependencies import _safe_import

torch = _safe_import("torch")
nn = _safe_import("torch.nn", return_object="None")
F = _safe_import("torch.nn.functional")
PretrainedConfig = _safe_import("transformers.PretrainedConfig")
```

`_safe_import` returns a mock when the package is missing, so base classes such as
`nn.Module` work at import time. It doesn't work as the parent of a `dataclass`. Look at
`sktime/libs/kronos/module.py` for `nn.Module` usage. If a file can't be made importable,
guard it in `__init__.py` so it's only imported when the dependency exists.

Vendored code may need exceptions in the ruff config: see `[tool.ruff.lint.per-file-ignores]`
in `pyproject.toml`, which has entries for other vendored libraries.

## Pin to the versions it works with

Vendored code isn't a first-class citizen of `transformers`, so it's likely to break on some
versions, especially around `generate`, `dtype`/`torch_dtype` and `PreTrainedModel` internals.

- Find the true lower bound by trying older versions, and put it in `python_dependencies`.
  The latest release is not a lower bound.
- Run the **real checkpoint** at that bound. `SundialForecaster` (#11349) couldn't load a
  checkpoint under its own pin, because the code used `dtype=` where that version wanted
  `torch_dtype=`.
- Check that the bound doesn't exclude a Python version sktime supports. Recent
  `transformers` releases need 3.11+, while sktime supports 3.10.
- Widening support to more versions is good only if it's cheap.

If you can't vendor and have to import private upstream modules, pin the dependency and add
a comment explaining why, since those modules can change in any release.

## Register it

- Add `"tests:libs": ["sktime.libs.<model>"]` to the forecaster's tags, so changes to the
  fork trigger its tests.
- Add an entry in `sktime/libs/README.md`, under "private vendor forks in `sktime`", with
  the source and license.

## Changes from upstream

Keep the fork close to upstream. When you do change behaviour, e.g. to fix an upstream bug,
keep upstream's behaviour reachable behind a flag that defaults to it (see
`adapter-checklist.md`, Constructor parameters), and list each change in a "Changes From
Upstream" section of the vendored `README.md`, as `sktime/libs/tempo/README.md` does.
