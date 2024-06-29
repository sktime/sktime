"""Tests for tabular-to-panel distance aggregation/reduction."""

import numpy as np
import pytest

from sktime.datatypes import convert_to
from sktime.dists_kernels.compose_tab_to_panel import AggrDist
from sktime.dists_kernels.scipy_dist import ScipyDist
from sktime.registry import all_estimators
from sktime.tests.test_switch import run_test_module_changed
from sktime.utils._testing.panel import make_transformer_problem


@pytest.fixture
def pw_trafo_tab():
    yield from all_estimators(
        estimator_types="transformer-pairwise", return_names=False
    )


AGGFUNCS = [
    np.mean,
    np.std,
    np.var,
    np.sum,
    np.prod,
    np.min,
    np.max,
    np.argmin,
    np.argmax,
    np.any,
]


@pytest.fixture(params=["list_df", "num_pan"])
def X1_X2(request):
    X1_list_df = make_transformer_problem(
        n_instances=4, n_columns=4, n_timepoints=5, random_state=1, return_numpy=False
    )
    X2_list_df = make_transformer_problem(
        n_instances=5, n_columns=4, n_timepoints=5, random_state=2, return_numpy=False
    )

    X1_num_pan = convert_to(X1_list_df, to_type="numpy3D")
    X2_num_pan = convert_to(X2_list_df, to_type="numpy3D")

    if request.param == "list_df":
        return (X1_list_df, X2_list_df)
    elif request.param == "num_pan":
        return (X1_num_pan, X2_num_pan)


@pytest.mark.skipif(
    not run_test_module_changed("sktime.dists_kernels"),
    reason="run test only if softdeps are present and incrementally (if requested)",
)
@pytest.mark.parametrize("aggfunc", AGGFUNCS)
def test_aggr(X1_X2, pw_trafo_tab, aggfunc):
    """Test that AggrDist produces expected pre-computed result on fixtures."""
    x, y = X1_X2
    transformer = pw_trafo_tab

    # default parametersc
    default_params = AggrDist(transformer=ScipyDist())
    default_params_transformation = np.around(
        (default_params.transform(x, y)), decimals=3
    )

    assert np.array_equal(
        np.array(
            [
                [1.714, 1.49, 1.53, 1.699, 1.849],
                [1.479, 1.36, 1.358, 1.476, 1.471],
                [1.553, 1.476, 1.354, 1.523, 1.425],
                [1.641, 1.704, 1.603, 1.698, 1.37],
            ]
        ),
        default_params_transformation,
    ), "Error occurred testing on default parameters, result is not correct"

    aggfunc_params = AggrDist(transformer=transformer(), aggfunc=aggfunc)
    aggfunc_params_transformation = aggfunc_params.transform(x, y)
    assert isinstance(aggfunc_params_transformation, np.ndarray), (
        f"Error occurred testing on following parameters"
        f"transformer={transformer}, aggfunc={aggfunc}"
    )
