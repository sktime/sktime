"""Cython-based MiniRocket prototype using compile-on-demand."""

import numpy as np
import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.dependencies.compile import import_or_compile_extension

# We will import this dynamically inside the class methods when needed.
_kernel = None


def _get_kernel():
    global _kernel
    if _kernel is None:
        _kernel = import_or_compile_extension(
            "sktime.transformations.rocket._rocket_cython_kernel",
            "_rocket_cython_kernel.pyx",
            __file__,
        )
    return _kernel


class MiniRocketCython(BaseTransformer):
    """MiniRocket prototype using dynamic Cython compilation on-demand."""

    _tags = {
        "authors": ["purvanshjoshi"],
        "maintainers": ["purvanshjoshi"],
        "requires_cython": True,
        "python_dependencies": "cython",
        "tests:vm": True,
        "capability:multivariate": False,
        "fit_is_empty": True,
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Primitives",
        "scitype:instancewise": True,
        "X_inner_mtype": "numpy3D",
        "y_inner_mtype": "None",
    }

    def __init__(self, scalar=2.0):
        self.scalar = scalar
        super().__init__()

    def _transform(self, X, y=None):
        # X shape: [n_instances, n_dimensions, series_length]
        n_instances, _, series_length = X.shape
        X = X.astype(np.float32)

        # Call the compiled Cython kernel for each instance
        kernel = _get_kernel()
        results = []
        for i in range(n_instances):
            # Flatten to 1D for simplicity in POC
            val = kernel.multiply_accumulate(X[i, 0, :], self.scalar)
            results.append(val)

        return pd.DataFrame(np.array(results).reshape(-1, 1))
