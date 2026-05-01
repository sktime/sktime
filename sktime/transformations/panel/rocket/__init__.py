# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Backward-compat alias for rocket transformers.

This package forwards to the flat layout. New code should import from
``sktime.transformations`` directly (e.g.
``from sktime.transformations.rocket import Rocket``).
"""

import importlib
import sys
import warnings

from sktime.transformations.minirocket import MiniRocket  # noqa: F401
from sktime.transformations.minirocket_multivariate import (  # noqa: F401
    MiniRocketMultivariate,
)
from sktime.transformations.minirocket_multivariate_variable import (  # noqa: F401
    MiniRocketMultivariateVariable,
)
from sktime.transformations.multirocket import MultiRocket  # noqa: F401
from sktime.transformations.multirocket_multivariate import (  # noqa: F401
    MultiRocketMultivariate,
)
from sktime.transformations.rocket import Rocket  # noqa: F401
from sktime.transformations.rocket_pyts import RocketPyts  # noqa: F401

__all__ = [
    "Rocket",
    "RocketPyts",
    "MiniRocket",
    "MiniRocketMultivariate",
    "MiniRocketMultivariateVariable",
    "MultiRocket",
    "MultiRocketMultivariate",
]

# Make legacy submodule paths resolve to the new flat modules.
# All were private files; aliased only for pickle compat.
_legacy_to_new = {
    "_rocket": "sktime.transformations.rocket",
    "_rocket_pyts": "sktime.transformations.rocket_pyts",
    "_minirocket": "sktime.transformations.minirocket",
    "_minirocket_multivariate": "sktime.transformations.minirocket_multivariate",
    "_minirocket_multivariate_variable": "sktime.transformations.minirocket_multivariate_variable",
    "_multirocket": "sktime.transformations.multirocket",
    "_multirocket_multivariate": "sktime.transformations.multirocket_multivariate",
    "_rocket_numba": "sktime.transformations._rocket_numba",
    "_minirocket_numba": "sktime.transformations._minirocket_numba",
    "_minirocket_multi_numba": "sktime.transformations._minirocket_multi_numba",
    "_minirocket_multi_var_numba": "sktime.transformations._minirocket_multi_var_numba",
    "_multirocket_numba": "sktime.transformations._multirocket_numba",
    "_multirocket_multi_numba": "sktime.transformations._multirocket_multi_numba",
}
for _legacy, _new in _legacy_to_new.items():
    sys.modules[f"{__name__}.{_legacy}"] = importlib.import_module(_new)

warnings.warn(
    "sktime.transformations.panel.rocket is deprecated and will be removed in "
    "a future release. Import from sktime.transformations directly "
    "(e.g. sktime.transformations.rocket.Rocket).",
    DeprecationWarning,
    stacklevel=2,
)
