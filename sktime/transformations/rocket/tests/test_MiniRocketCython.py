"""Tests for MiniRocketCython prototype."""

import numpy as np
import pytest

from sktime.transformations.rocket._rocket_cython import MiniRocketCython

try:
    from sktime.utils.dependencies.compile import has_compiler
except ImportError:
    pytest.skip("compile module not available", allow_module_level=True)


def test_minirocket_cython_compile_and_transform():
    """Verify MiniRocketCython compiles/transforms, or fails cleanly."""
    # 3 instances, 1 dimension, 5 timepoints
    X = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0, 5.0]],
            [[2.0, 4.0, 6.0, 8.0, 10.0]],
            [[5.0, 5.0, 5.0, 5.0, 5.0]],
        ],
        dtype=np.float32,
    )

    transformer = MiniRocketCython(scalar=2.0)

    if not has_compiler():
        # Verify raises ImportError when compiler toolchain is missing
        expected_errs = (ImportError, ModuleNotFoundError)
        match_msg = "C compiler is required|No working C compiler"
        with pytest.raises(expected_errs, match=match_msg):
            transformer.fit_transform(X)
    else:
        res = transformer.fit_transform(X)

        # Expected values:
        # Instance 1: (1+2+3+4+5) * 2 = 15 * 2 = 30
        # Instance 2: (2+4+6+8+10) * 2 = 30 * 2 = 60
        # Instance 3: (5+5+5+5+5) * 2 = 25 * 2 = 50
        expected = np.array([[30.0], [60.0], [50.0]], dtype=np.float32)

        np.testing.assert_array_almost_equal(res.values, expected)
