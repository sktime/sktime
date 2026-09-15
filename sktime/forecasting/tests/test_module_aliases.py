#!/usr/bin/env python3 -u
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Tests for the deprecated sktime.forecasting submodule alias mechanism.

See bug report #11027: importing sktime.forecasting used to eagerly import
every module in _MODULE_ALIASES, including torch-based ones, causing high
memory usage (especially with several pytest-xdist workers doing this at
once). The aliases must stay resolvable, but only load their real, possibly
heavy, target module on first use.

Each check below runs in its own subprocess: this keeps the eager-import
check honest (no other test in the same process may have already imported
the target module first) and avoids repeatedly importing several different
torch-based modules within one interpreter, which is independently fragile
for reasons unrelated to this alias mechanism.
"""

import subprocess
import sys

from sktime.forecasting import _MODULE_ALIASES

__author__ = ["felipebridge"]

# boxcox_bias_adjusted_forecaster -> boxcox_biasadj has no heavy soft
# dependencies, so it is used as the representative alias for the
# functional (still-resolves) checks; the mechanism itself is identical
# for every entry in _MODULE_ALIASES.
_ALIAS_NAME, _TARGET_NAME = next(iter(_MODULE_ALIASES.items()))


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(  # noqa: S603
        [sys.executable, "-c", code], capture_output=True, text=True
    )


def test_importing_forecasting_does_not_eagerly_load_alias_targets():
    """Importing sktime.forecasting must not import the aliased target modules."""
    result = _run(
        "import sys\n"
        "import sktime.forecasting\n"
        f"assert 'sktime.forecasting.{_TARGET_NAME}' not in sys.modules, "
        f"'sktime.forecasting.{_TARGET_NAME} was eagerly imported'\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_deprecated_submodule_import_still_resolves():
    """Old sktime.forecasting.<alias_name> submodule imports must still work."""
    result = _run(
        "import warnings\n"
        "import importlib\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        f"    module = importlib.import_module('sktime.forecasting.{_ALIAS_NAME}')\n"
        "    module.BoxCoxBiasAdjustedForecaster\n"
        "assert any(issubclass(w.category, FutureWarning) for w in caught), caught\n"
        f"assert any('{_TARGET_NAME}' in str(w.message) for w in caught), caught\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_deprecated_attribute_access_still_resolves():
    """Old sktime.forecasting.<alias_name> attribute access must still work."""
    result = _run(
        "import warnings\n"
        "import sktime.forecasting\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always')\n"
        f"    module = getattr(sktime.forecasting, '{_ALIAS_NAME}')\n"
        f"expected = 'sktime.forecasting.{_TARGET_NAME}'\n"
        "assert module.__name__ == expected, module.__name__\n"
        "assert any(issubclass(w.category, FutureWarning) for w in caught), caught\n"
        "print('OK')\n"
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout
