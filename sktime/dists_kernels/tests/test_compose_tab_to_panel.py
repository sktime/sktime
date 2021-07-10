# -*- coding: utf-8 -*-
import numpy as np
from sktime.dists_kernels.compose_tab_to_panel import AggrDist
from sktime.dists_kernels.scipy_dist import ScipyDist
from sktime.utils._testing.panel import make_transformer_problem
from sktime.utils import all_estimators

PAIRWISE_TRANSFORMERS_TAB = all_estimators(
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


X1_num_pan = make_transformer_problem(
    n_instances=4, n_columns=4, n_timepoints=5, random_state=1, return_numpy=True
)
X2_num_pan = make_transformer_problem(
    n_instances=5, n_columns=5, n_timepoints=5, random_state=2, return_numpy=True
)

X1_list_df = make_transformer_problem(
    n_instances=4, n_columns=4, n_timepoints=5, random_state=1, return_numpy=False
)
X2_list_df = make_transformer_problem(
    n_instances=5, n_columns=5, n_timepoints=5, random_state=2, return_numpy=False
)

X1_num_df = np.array(X1_list_df)
X2_num_df = np.array(X2_list_df)


def test_aggr():
    # test 3d numpy
    _run_aggr_dist_test(X1_num_pan, X2_num_pan)

    # test list of df
    _run_aggr_dist_test(X1_list_df, X2_list_df)

    # test numpy of df
    _run_aggr_dist_test(X1_num_df, X2_num_df)


def _run_aggr_dist_test(x, y):
    # default parameters
    default_params = AggrDist(transformer=ScipyDist())
    default_params_transformation = np.around(
        (default_params.transform(x, y)), decimals=3
    )

    assert np.array_equal(
        np.array(
            [
                [1.649, 1.525, 1.518, 1.934, 1.636],
                [1.39, 1.395, 1.313, 1.617, 1.418],
                [1.492, 1.489, 1.323, 1.561, 1.437],
                [1.559, 1.721, 1.544, 1.589, 1.642],
            ]
        ),
        default_params_transformation,
    ), "Error occurred testing on default parameters, result is not correct"

    for transformer in PAIRWISE_TRANSFORMERS_TAB:
        for aggfunc in AGGFUNCS:
            aggfunc_params = AggrDist(transformer=transformer(), aggfunc=aggfunc)
            aggfunc_params_transformation = aggfunc_params.transform(x, y)
            assert isinstance(aggfunc_params_transformation, np.ndarray), (
                f"Error occurred testing on following parameters"
                f"transformer={transformer}, aggfunc={aggfunc}"
            )
