# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Kalman Filter transformers unit tests."""

__author__ = ["NoaBenAmi"]

import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

from sktime.transformations.series.kalman_filter import (
    KalmanFilterTransformerFP,
    KalmanFilterTransformerPK,
)
from sktime.utils.validation._dependencies import _check_soft_dependencies

# ts stands for time steps
ts = 10


def create_data(shape, missing_values=False, p=0.15, mult=10):
    """Create random ndarray of shape `shape`.

    The result array will contain missing values (represented by np.nan)
    if parameter `missing_values` is set to true.
    """
    if isinstance(shape, int):
        shape = (shape,)
    data = np.random.rand(*shape) * mult

    if missing_values:
        time_steps = shape[0]
        measurement_dim = shape[1]
        pr = max(int(time_steps * p), 4)
        for t in range(time_steps):
            if t % pr == 0:
                data[t] = [np.nan] * measurement_dim
    return data


def rand_list(shape, length=ts, mult=10):
    """Return list with `length` random ndarrays of shape `shape`."""
    return [create_data(shape, mult=mult) for _ in range(length)]


# state_dim = 3, measurement_dim = 3, time_steps = ts (10)
params_3_3_dynamic = {
    "process_noise": create_data((ts, 3, 3)),  # random ndarray of shape (ts, 3, 3),
    "measurement_noise": rand_list(
        (3, 3), length=ts
    ),  # [`ts` random ndarrays of shape (3, 3)],
    "measurement_function": create_data((3, 3)),  # random ndarray of shape (3, 3),
    "initial_state_covariance": create_data((3, 3)),  # random ndarray of shape (3, 3),
}

# state_dim = 3, measurement_dim = 3, time_steps = ts (10)
params_3_3_static = {
    "state_transition": create_data((3, 3)),  # random ndarray of shape (3,3),
    "process_noise": create_data((3, 3)),  # random ndarray of shape (3,3),
    "measurement_noise": create_data((3, 3)),  # random ndarray of shape (3,3),
    "initial_state": create_data(3),  # random ndarray of shape (3,),
    "initial_state_covariance": create_data((3, 3)),  # random ndarray of shape (3,3)
}

# state_dim = 2, measurement_dim = 3, time_steps = ts (10)
params_2_3_ = {
    "state_transition": rand_list(
        (2, 2), length=ts
    ),  # [`ts` random ndarrays of shape (2,2)],
    "measurement_noise": create_data((3, 3)),  # random ndarray of shape (3, 3),
    "measurement_function": rand_list(
        (3, 2), length=ts
    ),  # [`ts` random ndarrays of shape (3,2)],
    "initial_state": create_data(2),  # random ndarray of shape (2,),
    "initial_state_covariance": create_data((2, 2)),  # random ndarray of shape (2,2)
}

# state_dim = 1, measurement_dim = 1, time_steps = ts (10)
params_1_1_arrays = {
    "state_transition": create_data((ts, 1, 1)),  # random ndarray of shape (ts, 1, 1)
    "process_noise": create_data((ts, 1, 1)),  # random ndarray of shape (ts, 1, 1)
    "initial_state": create_data(1),  # random ndarray of shape (1,)
    "initial_state_covariance": create_data((1, 1)),  # random ndarray of shape (1, 1)
}

# state_dim = 1, measurement_dim = 1, time_steps = ts (10)
params_1_1_lists = {
    "process_noise": rand_list(
        (1, 1), length=ts
    ),  # [`ts` random ndarrays of shape (1,1)],
    "measurement_noise": rand_list(
        (1, 1), length=ts
    ),  # [`ts` random ndarrays of shape (1,1)],
    "measurement_function": rand_list(
        (1, 1), length=ts
    ),  # [`ts` random ndarrays of shape (1,1)],
    "initial_state": create_data(1),  # random ndarray of shape (1,)
}

# state_dim = 3, measurement_dim = 1, time_steps = ts (10)
params_3_1_lists = {
    "state_transition": rand_list(
        (3, 3), length=ts
    ),  # [`ts` random ndarrays of shape (3,3)],
    "process_noise": rand_list(
        (3, 3), length=ts
    ),  # [`ts` random ndarrays of shape (3,3)],
    "measurement_noise": rand_list(
        (1, 1), length=ts
    ),  # [`ts` random ndarrays of shape (1,1)],
    "measurement_function": rand_list(
        (1, 3), length=ts
    ),  # [`ts` random ndarrays of shape (1,3)],
}


def get_params_mapping(params):
    """Transform parameters names.

    From KalmanFilterTransformerPK, KalmanFilterTransformerFP naming
    forms to `pykalman`'s naming form.
    """
    params_mapping = {
        "state_transition": "transition_matrices",
        "process_noise": "transition_covariance",
        "measurement_offsets": "observation_offsets",
        "transition_offsets": "transition_offsets",
        "measurement_noise": "observation_covariance",
        "measurement_function": "observation_matrices",
        "initial_state": "initial_state_mean",
        "initial_state_covariance": "initial_state_covariance",
    }
    if params is None:
        return None
    if isinstance(params, str) and params == "all":
        return list(params_mapping.values())
    return [params_mapping[param] for param in params]


def init_kf_pykalman(
    state_dim,
    measurement_dim=None,
    state_transition=None,
    process_noise=None,
    measurement_noise=None,
    measurement_function=None,
    initial_state=None,
    initial_state_covariance=None,
    transition_offsets=None,
    measurement_offsets=None,
    estimate_matrices=None,
    denoising=False,
):
    """Initiate instance of `pykalman`'s `KalmanFilter`."""
    from pykalman.standard import KalmanFilter

    em_vars = get_params_mapping(params=estimate_matrices)
    kf_pykalman = KalmanFilter(
        transition_matrices=state_transition,
        observation_matrices=measurement_function,
        transition_covariance=process_noise,
        observation_covariance=measurement_noise,
        transition_offsets=transition_offsets,
        observation_offsets=measurement_offsets,
        initial_state_mean=initial_state,
        initial_state_covariance=initial_state_covariance,
        n_dim_state=state_dim,
        n_dim_obs=measurement_dim,
        em_vars=em_vars,
    )
    return kf_pykalman


def init_kf_filterpy(measurements, adapter, n=10, y=None):
    """Adjust params and measurements.

    Given measurements and adapter, adjust params and measurements to
    `FilterPy` usable form.
    """
    y_dim = 1 if y is None else y.shape[-1]

    G = (
        np.eye(adapter.state_dim, y_dim)
        if adapter.control_transition is None
        else np.atleast_2d(adapter.control_transition)
    )

    matrices = {
        "Fs": [adapter.F_] * n if adapter.F_.ndim == 2 else [f for f in adapter.F_],
        "Qs": [adapter.Q_] * n if adapter.Q_.ndim == 2 else [q for q in adapter.Q_],
        "Rs": [adapter.R_] * n if adapter.R_.ndim == 2 else [r for r in adapter.R_],
        "Hs": [adapter.H_] * n if adapter.H_.ndim == 2 else [h for h in adapter.H_],
        "Bs": [G] * n if G.ndim == 2 else [g for g in G],
        "us": None if y is None else ([y] * n if y.ndim == 1 else [_y for _y in y]),
        "x": adapter.X0_,
        "P": adapter.P0_,
    }
    data = [None if any(np.isnan(d)) else d.copy() for d in measurements]

    return matrices, data


@pytest.mark.skipif(
    not _check_soft_dependencies("pykalman", severity="none"),
    reason="skip test if required soft dependency pykalman not available",
)
@pytest.mark.parametrize(
    "params, measurements",
    [  # test case 1 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_dynamic
        (dict(params_3_3_dynamic, state_dim=3), create_data((ts, 3))),
        # H and X0 will be estimated using em algorithm.
        (
            dict(
                params_3_3_dynamic,
                state_dim=3,
                estimate_matrices=["measurement_function", "initial_state"],
            ),
            create_data((ts, 3)),
        ),
        # test case 2 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_static
        (dict(params_3_3_static, state_dim=3), create_data((ts, 3))),
        # all matrix parameters will be estimated using em algorithm.
        (
            dict(params_3_3_static, state_dim=3, estimate_matrices="all"),
            create_data((ts, 3)),
        ),
        # test case 3 -
        #   state_dim = 2, measurement_dim = 3, params - params_2_3_,
        #   b is set with random ndarray of shape (2,).
        (
            dict(params_2_3_, state_dim=2, transition_offsets=create_data(2)),
            create_data((ts, 3)),
        ),
        # b, d and R will be estimated using em algorithm.
        (
            dict(
                params_2_3_,
                state_dim=2,
                transition_offsets=create_data(2),
                estimate_matrices=[
                    "measurement_offsets",
                    "transition_offsets",
                    "measurement_noise",
                ],
            ),
            create_data((ts, 3)),
        ),
        # test 4 -
        #   state_dim = 3, measurement_dim = 1, params are None
        (dict(state_dim=3), create_data((ts, 1), missing_values=True)),
        # F and Q will be estimated using em algorithm.
        (
            dict(state_dim=3, estimate_matrices=["state_transition", "process_noise"]),
            create_data((ts, 1), missing_values=True),
        ),
        # test 5 -
        #   state_dim = 1, measurement_dim = 1, params - params_1_1_arrays,
        #   b and d each set with random ndarray of shape (1,).
        (
            dict(
                params_1_1_arrays,
                state_dim=1,
                transition_offsets=create_data(1),
                measurement_offsets=create_data(1),
            ),
            create_data((ts, 1)),
        ),
        # P0 will be estimated using em algorithm.
        (
            dict(
                params_1_1_arrays,
                state_dim=1,
                estimate_matrices=["initial_state_covariance"],
                transition_offsets=create_data(1),
                measurement_offsets=create_data(1),
            ),
            create_data((ts, 1)),
        ),
        # test case 6 -
        #   state_dim = 1, measurement_dim = 1, params - params_1_1_lists
        #   b and d each set with a list of random ndarrays.
        #   transition_offsets (aka b) - each element of b is of shape
        #   (1,), b.length == 10.
        #   measurement_offsets (aka d) - each element of d is of shape
        #   (1,), d.length == 10.
        (
            dict(
                params_1_1_lists,
                state_dim=1,
                transition_offsets=rand_list(1),
                measurement_offsets=rand_list(1),
            ),
            create_data((ts, 1), missing_values=True),
        ),
        # test case 7 -
        #   state_dim = 3, measurement_dim = 1, params - params_3_1_lists
        #   b and d each set with a list of random ndarrays.
        #   transition_offsets (aka b) - each element of b is of shape
        #   (3,), b.length == 10.
        #   measurement_offsets (aka d) - each element of d is of shape
        #   (1,), d.length == 10.
        (
            dict(
                params_3_1_lists,
                state_dim=3,
                transition_offsets=rand_list(3),
                measurement_offsets=rand_list(1),
            ),
            create_data((ts, 1), missing_values=True),
        ),
    ],
)
def test_transform_and_smooth_pk(params, measurements):
    """Test KalmanFilterTransformerPK `fit` and `transform`.

    Creating two instances of KalmanFilterTransformerPK, one instance
    with parameter `denoising` set to False, and the other's set to True.
    Compare result with `pykalman`'s `filter` and `smooth`.
    """
    mask_measurements = np.ma.masked_invalid(np.copy(measurements))

    # adapter transformer
    adapter_transformer = KalmanFilterTransformerPK(
        **params
    )  # init_pykalman_adapter(state_dim=state_dim, **params)
    xt_adapter_transformer = adapter_transformer.fit_transform(measurements)

    # adapter denoising
    adapter_denoising = KalmanFilterTransformerPK(**params, denoising=True)
    xt_adapter_denoising = adapter_denoising.fit_transform(measurements)

    # pykalman
    kf_pykalman = init_kf_pykalman(measurement_dim=measurements.shape[1], **params)
    if "estimate_matrices" in params.keys():
        kf_pykalman = kf_pykalman.em(mask_measurements)
    xt_pykalman_transformer, _ = kf_pykalman.filter(mask_measurements)
    xt_pykalman_denoising, _ = kf_pykalman.smooth(mask_measurements)

    # test filter()
    assert np.array_equal(xt_adapter_transformer, xt_pykalman_transformer)

    # test smooth()
    assert np.array_equal(xt_adapter_denoising, xt_pykalman_denoising)


@pytest.mark.skipif(
    not _check_soft_dependencies("pykalman", "filterpy", severity="none"),
    reason="skip test if required soft dependencies pykalman, filterpy not available",
)
@pytest.mark.parametrize(
    "classes, params, measurements",
    [  # test case 1 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_dynamic,
        #   X0 and H will be estimated using em algorithm.
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(
                params_3_3_dynamic,
                state_dim=3,
                estimate_matrices=["measurement_function", "initial_state"],
            ),
            create_data((ts, 3)),
        ),
        # test case 2 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_static,
        #   all matrix parameters will be estimated using em algorithm.
        # test pykalman, transition_offsets (aka b) - is set with
        # random ndarray of shape (3,).
        (
            [KalmanFilterTransformerPK],
            dict(
                params_3_3_static,
                state_dim=3,
                transition_offsets=create_data(3),
                estimate_matrices=[
                    "state_transition",
                    "process_noise",
                    "measurement_offsets",
                    "transition_offsets",
                    "measurement_noise",
                    "measurement_function",
                    "initial_state",
                    "initial_state_covariance",
                ],
            ),
            create_data((ts, 3), missing_values=True),
        ),
        # test FilterPy
        (
            [KalmanFilterTransformerFP],
            dict(
                params_3_3_static,
                state_dim=3,
                estimate_matrices=[
                    "state_transition",
                    "process_noise",
                    "measurement_noise",
                    "measurement_function",
                    "initial_state",
                    "initial_state_covariance",
                ],
            ),
            create_data((ts, 3), missing_values=True),
        ),
        # test case 3 -
        #   state_dim = 2, measurement_dim = 3, params - params_2_3_
        # test both adapters, matrices Q, R, X0, P0 are estimated using em algorithm.
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(
                params_2_3_,
                state_dim=2,
                estimate_matrices=[
                    "process_noise",
                    "measurement_noise",
                    "initial_state",
                    "initial_state_covariance",
                ],
            ),
            create_data((ts, 3)),
        ),
        # test pykalman, transition_offsets (aka b) - is set with
        # random ndarray of shape (2,).
        (
            [KalmanFilterTransformerPK],
            dict(
                params_2_3_,
                state_dim=2,
                transition_offsets=create_data(2),
                estimate_matrices=["measurement_offsets", "transition_offsets"],
            ),
            create_data((ts, 3)),
        ),
        # test case 4 -
        #   state_dim = 3, measurement_dim = 1, params are None,
        #   all matrix parameters will be estimated using em algorithm.
        #   transition_offsets (aka b) - set with random ndarray of shape (3,).
        #   measurement_offsets (aka d) - set with random ndarray of shape (1,).
        # test pykalman
        (
            [KalmanFilterTransformerPK],
            dict(
                state_dim=3,
                transition_offsets=create_data(3),
                measurement_offsets=create_data(1),
                estimate_matrices="all",
            ),
            create_data((ts, 1), missing_values=True),
        ),
        # test case 5 -
        #   state_dim = 2, measurement_dim = 4, params are None,
        #   H and Q will be estimated using em algorithm.
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(
                state_dim=2, estimate_matrices=["measurement_function", "process_noise"]
            ),
            create_data((ts, 4), missing_values=True),
        ),
        # test case 6 -
        #   state_dim = 1, measurement_dim = 1, params - params_1_1_arrays
        # test pykalman, b and d each set with random ndarray of shape (1,).
        # b, d, X0, P0 will be estimated using em algorithm.
        (
            [KalmanFilterTransformerPK],
            dict(
                params_1_1_arrays,
                state_dim=1,
                transition_offsets=create_data(1),
                measurement_offsets=create_data(1),
                estimate_matrices=[
                    "measurement_offsets",
                    "transition_offsets",
                    "initial_state",
                    "initial_state_covariance",
                ],
            ),
            create_data((ts, 1)),
        ),
        # test pykalman, b and d each set with random ndarray of shape (10, 1).
        # R and H will be estimated using em algorithm.
        (
            [KalmanFilterTransformerPK],
            dict(
                params_1_1_arrays,
                state_dim=1,
                transition_offsets=create_data((ts, 1)),
                measurement_offsets=create_data((ts, 1)),
                estimate_matrices=["measurement_noise", "measurement_function"],
            ),
            create_data((ts, 1)),
        ),
        # test FilterPy, R, H, X0 and P0 will be estimated using em algorithm.
        (
            [KalmanFilterTransformerFP],
            dict(
                params_1_1_arrays,
                state_dim=1,
                estimate_matrices=[
                    "measurement_noise",
                    "measurement_function",
                    "initial_state",
                    "initial_state_covariance",
                ],
            ),
            create_data((ts, 1)),
        ),
        # test case 7 -
        #   state_dim = 1, measurement_dim = 1, params - params_1_1_lists,
        #   F will be estimated using em algorithm.
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(params_1_1_lists, state_dim=1, estimate_matrices=["state_transition"]),
            create_data((ts, 1), missing_values=True),
        ),
        # test case 8 -
        #   state_dim = 3, measurement_dim = 1, params - params_3_1_lists,
        #   X0 and P0 will be estimated using em algorithm.
        # test pykalman, b and d each set with a list of random ndarrays.
        # transition_offsets (aka b) - each element of b is of shape (3,),
        # b.length == 10.
        # measurement_offsets (aka d) - each element of d is of shape (1,),
        # d.length == 10.
        (
            [KalmanFilterTransformerPK],
            dict(
                params_3_1_lists,
                state_dim=3,
                transition_offsets=rand_list(3),
                measurement_offsets=rand_list(1),
                estimate_matrices=["initial_state", "initial_state_covariance"],
            ),
            create_data((ts, 1)),
        ),
        # test FilterPy
        (
            [KalmanFilterTransformerFP],
            dict(
                params_3_1_lists,
                state_dim=3,
                estimate_matrices=["initial_state", "initial_state_covariance"],
            ),
            create_data((ts, 1)),
        ),
    ],
)
def test_em(classes, params, measurements):
    """Test adapters matrix estimation.

    Call `fit` of input adapter/s, and compare all matrix parameters
    with `pykalman`'s matrix parameters returned from `em`.
    This test is useful for both KalmanFilterTransformerPK and
    KalmanFilterTransformerFP.
    """
    mask_measurements = np.ma.masked_invalid(np.copy(measurements))

    for _class in classes:
        adapter = _class(**params)
        adapter = adapter.fit(measurements)

        kf_pykalman = init_kf_pykalman(measurement_dim=measurements.shape[1], **params)
        kf_pykalman = kf_pykalman.em(X=mask_measurements)

        assert np.array_equal(adapter.F_, kf_pykalman.transition_matrices)
        assert np.array_equal(adapter.H_, kf_pykalman.observation_matrices)
        assert np.array_equal(adapter.Q_, kf_pykalman.transition_covariance)
        assert np.array_equal(adapter.R_, kf_pykalman.observation_covariance)
        assert np.array_equal(adapter.X0_, kf_pykalman.initial_state_mean)
        assert np.array_equal(adapter.P0_, kf_pykalman.initial_state_covariance)

        if hasattr(adapter, "transition_offsets_"):
            assert np.array_equal(
                adapter.transition_offsets_, kf_pykalman.transition_offsets
            )
        if hasattr(adapter, "measurement_offsets_"):
            assert np.array_equal(
                adapter.measurement_offsets_, kf_pykalman.observation_offsets
            )


@pytest.mark.skipif(
    not _check_soft_dependencies("pykalman", "filterpy", severity="none"),
    reason="skip test if required soft dependencies pykalman, filterpy not available",
)
@pytest.mark.parametrize(
    "classes, params, measurements",
    [  # test case 1 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_dynamic
        # bad input :
        # typo in element of `estimate_matrices` : sstate_transition instead of
        # state_transition
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(
                params_3_3_dynamic,
                state_dim=3,
                estimate_matrices=["sstate_transition", "process_noise"],
            ),
            create_data((ts, 3)),
        ),
        # test case 2 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_dynamic
        # bad input:
        # wrong b shape: set to ndarray of shape (10, 2) when should be (10, 3) or (3,)
        (
            [KalmanFilterTransformerPK],
            dict(
                params_3_3_dynamic,
                state_dim=3,
                estimate_matrices=[
                    "process_noise",
                    "measurement_noise",
                    "initial_state",
                    "initial_state_covariance",
                ],
                transition_offsets=create_data((ts, 2)),
            ),
            create_data((ts, 3), missing_values=True),
        ),
        # test case 3 -
        #   state_dim = 2, measurement_dim = 3, params - params_2_3_
        # bad input:
        # wrong d shape: set to ndarray of shape (11, 2) when should be (10, 2) or (2,)
        (
            [KalmanFilterTransformerPK],
            dict(params_2_3_, state_dim=2, measurement_offsets=create_data((11, 2))),
            create_data((ts, 3)),
        ),
        # test case 4 -
        #   state_dim = 2, measurement_dim = 3, params - params_2_3_
        # bad input:
        # typo in element of `estimate_matrices` : transition_offset instead of
        # transition_offsets
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(params_2_3_, state_dim=2, estimate_matrices=["transition_offset"]),
            create_data((ts, 3)),
        ),
        # test case 5 -
        #   state_dim = 3, measurement_dim = 5, params - are None
        #   except state_transition.
        # bad input:
        # wrong F shape: set to ndarray of shape (10, 3, 1) when should be
        # (10, 3, 3) or (3, 3)
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(state_dim=3, state_transition=create_data((ts, 3, 1))),
            create_data((ts, 5), missing_values=True),
        ),
        # test case 6 -
        #   state_dim = 4, measurement_dim = 4, params - are None
        # bad input:
        # typo in element of `estimate_matrices` : measurement_functions instead
        # of measurement_function
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(state_dim=4, estimate_matrices=["measurement_functions"]),
            create_data((ts, 4), missing_values=True),
        ),
        # bad input:
        # KalmanFilterTransformerFP does not estimate matrix transition_offsets.
        (
            [KalmanFilterTransformerFP],
            dict(state_dim=4, estimate_matrices=["transition_offsets"]),
            create_data((ts, 4), missing_values=True),
        ),
        # bad input:
        # KalmanFilterTransformerFP does not estimate matrix measurement_offsets.
        (
            [KalmanFilterTransformerFP],
            dict(state_dim=4, estimate_matrices=["measurement_offsets"]),
            create_data((ts, 4), missing_values=True),
        ),
        # bad input:
        # KalmanFilterTransformerFP does not estimate matrix control_transition.
        (
            [KalmanFilterTransformerFP],
            dict(state_dim=4, estimate_matrices=["control_transition"]),
            create_data((ts, 4), missing_values=True),
        ),
        # test case 7 -
        #   state_dim = 1, measurement_dim = 1, params - params_1_1_arrays
        # bad input:
        # typo in element of `estimate_matrices` : initial_state_mean
        # instead of initial_state
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(
                params_1_1_arrays,
                state_dim=1,
                estimate_matrices=[
                    "state_transition",
                    "initial_state_mean",
                    "initial_state_covariance",
                ],
            ),
            create_data((ts, 1)),
        ),
        # test case 8 -
        #   state_dim = 2, measurement_dim = 3, params - as described
        # bad inputs:
        # typo in element of `estimate_matrices` : covariance instead of
        # initial_state_covariance
        # wrong H shape: set to ndarray of shape (2, 3) when should be
        # (10, 3, 2) or (3, 2)
        (
            [KalmanFilterTransformerPK, KalmanFilterTransformerFP],
            dict(
                state_dim=2,
                estimate_matrices=["measurement_noise", "covariance"],
                state_transition=create_data((3, 3)),
                measurement_function=create_data((2, 3)),
            ),
            create_data((ts, 3)),
        ),
    ],
)
def test_bad_inputs(classes, params, measurements):
    """Test adapters bad inputs error handling.

    Call `fit` of input adapter/s, and pass if ValueError
    was thrown.
    This test is useful for both KalmanFilterTransformerPK
    and KalmanFilterTransformerFP.
    """
    with pytest.raises(ValueError):
        for _class in classes:
            adapter = _class(**params)
            adapter.fit(X=measurements)


@pytest.mark.skipif(
    not _check_soft_dependencies("filterpy", severity="none"),
    reason="skip test if required soft dependency filterpy not available",
)
@pytest.mark.parametrize(
    "params, measurements, y",
    [  # test case 1 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_dynamic
        (dict(params_3_3_dynamic, state_dim=3), create_data((ts, 3)), None),
        # control_transition (aka G or B) is set with random ndarray of shape (3, 3).
        # y is set with random ndarray of shape (10, 3).
        # H and X0 will be estimated using em algorithm.
        (
            dict(
                params_3_3_dynamic,
                state_dim=3,
                control_transition=create_data((3, 3)),
                estimate_matrices=["measurement_function", "initial_state"],
            ),
            create_data((ts, 3)),
            create_data((ts, 3)),
        ),
        # test case 2 -
        #   state_dim = 3, measurement_dim = 3, params - params_3_3_static
        # control_transition (aka G or B) is set with a list of 10 random
        # ndarrays, each of shape (3, 2).
        # y is set with random ndarray of shape (2,).
        (
            dict(params_3_3_static, state_dim=3, control_transition=rand_list((3, 2))),
            create_data((ts, 3)),
            create_data(2),
        ),
        # all matrix parameters will be estimated using em algorithm.
        # control_transition (aka G or B) is set with random ndarray of shape (3, 1).
        # y is set with random ndarray of shape (1,).
        (
            dict(
                params_3_3_static,
                state_dim=3,
                estimate_matrices="all",
                control_transition=create_data((3, 1)),
            ),
            create_data((ts, 3)),
            create_data(1),
        ),
        # test case 3 -
        #   state_dim = 2, measurement_dim = 3, params - params_2_3_,
        # y is set with random ndarray of shape (10, 4).
        (dict(params_2_3_, state_dim=2), create_data((ts, 3)), create_data((ts, 4))),
        # R will be estimated using em algorithm.
        (
            dict(params_2_3_, state_dim=2, estimate_matrices=["measurement_noise"]),
            create_data((ts, 3)),
            None,
        ),
        # test case 4 -
        #   state_dim = 3, measurement_dim = 1, params are None
        # control_transition (aka G or B) - is set with random ndarray of shape (3, 2).
        # should raise a warning and control_transition will be
        # ignored during calculation.
        (
            dict(state_dim=3, control_transition=create_data((3, 2))),
            create_data((ts, 1), missing_values=True),
            None,
        ),
        # F and Q will be estimated using em algorithm.
        (
            dict(state_dim=3, estimate_matrices=["state_transition", "process_noise"]),
            create_data((ts, 1), missing_values=True),
            None,
        ),
        # test case 5 -
        #   state_dim = 1, measurement_dim = 1, params - params_1_1_arrays,
        # control_transition is set with random ndarray of shape (10, 1).
        # y is set with random ndarray of shape (10, 1).
        (
            dict(
                params_1_1_arrays,
                state_dim=1,
                control_transition=create_data((ts, 1, 1)),
            ),
            create_data((ts, 1)),
            create_data((ts, 1)),
        ),
        # P0 will be estimated using em algorithm.
        # control_transition is set with a list of 10 random ndarrays of shape (1,).
        # y is set with random ndarray of shape (10, 1).
        (
            dict(
                params_1_1_arrays,
                state_dim=1,
                estimate_matrices=["initial_state_covariance"],
                control_transition=rand_list((1, 1)),
            ),
            create_data((ts, 1)),
            create_data((ts, 1)),
        ),
        # test case 6 -
        #   state_dim = 1, measurement_dim = 1, params - params_1_1_lists
        # y is set with random ndarray of shape (2,).
        (
            dict(params_1_1_lists, state_dim=1),
            create_data((ts, 1), missing_values=True),
            create_data(2),
        ),
        # test case 7 -
        #   state_dim = 3, measurement_dim = 1, params - params_3_1_lists
        # control_transition is set with random ndarray of shape (3, 3).
        # y is set with random ndarray of shape (10, 1).
        (
            dict(params_3_1_lists, state_dim=3, control_transition=create_data((3, 3))),
            create_data((ts, 1), missing_values=True),
            create_data((ts, 3)),
        ),
    ],
)
def test_transform_and_smooth_fp(params, measurements, y):
    """Test KalmanFilterTransformerFP `fit` and `transform`.

    Creating two instances of KalmanFilterTransformerFP, one instance
    with parameter `denoising` set to False, and the other's set to True.
    Compare result with `FilterPy`'s `batch_filter` and `rts_smoother`.
    """
    from filterpy.kalman.kalman_filter import batch_filter, rts_smoother

    # initiate KalmanFilterTransformerFP with denoising=False
    # fit and transform
    adapter_transformer = KalmanFilterTransformerFP(**params)
    adapter_transformer = adapter_transformer.fit(measurements, y=y)
    xt_transformer_adapter = adapter_transformer.transform(measurements, y=y)

    # initiating KalmanFilterTransformerFP with denoising=True,
    # fit and transform
    adapter_smoother = KalmanFilterTransformerFP(denoising=True, **params)
    adapter_smoother = adapter_smoother.fit(measurements, y=y)
    xt_smoother_adapter = adapter_smoother.transform(measurements, y=y)

    # get data in a form compatible to FilterPy. call batch_filter
    matrices, fp_measurements = init_kf_filterpy(
        measurements=measurements, adapter=adapter_transformer, y=y
    )
    (means, covs, _, _) = batch_filter(**matrices, zs=fp_measurements)

    # test transformer
    assert_array_almost_equal(xt_transformer_adapter, means)

    # test smoother
    xt_smoother_filterpy = rts_smoother(
        Xs=means, Ps=covs, Fs=matrices["Fs"], Qs=matrices["Qs"]
    )[0]
    assert_array_almost_equal(xt_smoother_adapter, xt_smoother_filterpy)
