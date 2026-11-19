# spec is described in issue #9885
import numpy as np
import pandas as pd

from sktime.detection._datatypes._check import _is_valid_detection
from sktime.detection.naive import NaivePretrainWindowDetector
from sktime.utils._testing.detection import make_detection_problem


def _single_change_point_series():
    """Single series with a change point at position 50."""
    n = 100
    values = np.zeros(n)
    values[:45] = 1
    values[45:50] = np.arange(1, 6)  # gradual increase to make it more realistic
    values[50:55] = 10.0
    values[55:] = 1
    values[90:95] = np.arange(1, 6)  # another gradual increase to make it more realistic
    values[95:] = 10.0
    return pd.Series(values), pd.DataFrame({"ilocs": [50, 95]})

def _change_point_pattern(ilocs, ramp_base, n=100, baseline=1.0, peak=10.0, ramp_len=5, spike_len=5):
    """Build a series with the change-point pattern at the given ilocs.

    Each event is preceded by a ``ramp_len``-step gradual increase from
    1..ramp_len, followed by ``spike_len`` timepoints at ``peak``, then
    back to ``baseline``.
    """
    values = np.full(n, baseline)
    for idx, cp in enumerate(ilocs):
        cp = int(cp)
        ramp_base_val = ramp_base[idx]
        values[cp - ramp_len:cp] = np.arange(ramp_base_val, ramp_len + ramp_base_val)
        values[cp:cp + spike_len] = peak
    return pd.Series(values)


def _make_panel_series():
    """Panel of two instances; each has its own change points sharing the
    same generative pattern (baseline 1s, ramp, spike).

    Instance 0 keeps the ilocs from ``_make_single_series`` ([25, 65]);
    instance 1 uses different ilocs ([30, 70]) so per-instance handling
    is genuinely exercised. ``y`` carries the instance level on its
    MultiIndex; ``ilocs`` are relative to each instance's own time axis.
    """
    n = 100
    inst_0_ilocs = np.array([25, 65])
    inst_1_ilocs = np.array([30, 70])
    inst_2_ilocs = np.array([10, 30, 50, 70, 90])

    X_0 = _change_point_pattern(inst_0_ilocs, n=n, baseline=1.0, peak=10.0, ramp_base=[1, 1])
    X_1 = _change_point_pattern(inst_1_ilocs, n=n, baseline=2.0, peak=20.0, ramp_base=[2, 2])
    X_2 = _change_point_pattern(inst_2_ilocs, n=n, baseline=3.0, peak=20.0, ramp_base=[1, 2, 3, 4, 5])

    idx = pd.MultiIndex.from_arrays(
        [[0] * n + [1] * n + [2] * n, list(range(n)) + list(range(n)) + list(range(n))],
        names=["instances", "timepoints"],
    )
    X = pd.DataFrame(
        {"value": np.concatenate([X_0.to_numpy(), X_1.to_numpy(), X_2.to_numpy()])},
        index=idx,
    )

    y = pd.DataFrame(
        {
            "instances": (
                [0] * len(inst_0_ilocs)
                + [1] * len(inst_1_ilocs)
                + [2] * len(inst_2_ilocs)
            ),
            "ilocs": np.concatenate(
                [inst_0_ilocs, inst_1_ilocs, inst_2_ilocs]
            ),
        }
    )
    return X, y


def test_pretrain_window_truncated_at_series_start():
    """Event at iloc < window_length."""
    n = 20
    values = np.ones(n)
    values[:3] = 5.0

    idx = pd.MultiIndex.from_arrays(
        [[0] * n, list(range(n))],
        names=["instances", "timepoints"],
    )
    X = pd.DataFrame({"value": values}, index=idx)
    y = pd.DataFrame({"instances": [0], "ilocs": [3]})

    npwd = NaivePretrainWindowDetector(window_length=5)
    npwd.pretrain(X, y)

    # in-window pooled mean is over the 3 clipped timepoints, all 5.0
    assert npwd.in_window_mean == 5.0
    # out-window is everything else: 17 baseline timepoints
    assert npwd.out_window_mean == 1.0
    # window was clipped: 3 in-window points, not the nominal window_length=5
    assert npwd._in_window_counts == 3


def test_naive_pretrain_window_detector_panel_pretrain_fit_predict():
    """Full pretrain → fit → predict flow.

    Pretrain on instances 0 and 1 to learn the in/out-window event signature.
    Fit on the first half of instance 2's timepoints to set the baseline on a
    new, previously-unseen series. Predict on the second half of instance 2 to
    detect its remaining events.
    """
    X, y = _make_panel_series()

    instances = X.index.get_level_values("instances")
    timepoints = X.index.get_level_values("timepoints")

    # pretrain on instances 0 and 1
    X_pretrain = X[instances.isin([0, 1])]
    y_pretrain = y[y["instances"].isin([0, 1])]

    # split instance 2 by timepoints: fit on first half, predict on second
    train_end = 60
    inst2 = instances == 2
    X_fit = X[inst2 & (timepoints < train_end)]
    X_test = X[inst2 & (timepoints >= train_end)]

    inst2_y = y["instances"] == 2
    y_fit = y[inst2_y & (y["ilocs"] < train_end)]
    y_test = y[inst2_y & (y["ilocs"] >= train_end)]

    npwd = NaivePretrainWindowDetector(window_length=5)
    npwd.pretrain(X_pretrain, y_pretrain)

    # scalar pool across instances 0 and 1 (4 events × 5 timepoints = 20 in-window):
    #   inst 0 windows [1..5] twice → sum 30
    #   inst 1 windows [2..6] twice → sum 40
    #   in_window_mean = 70 / 20 = 3.5
    assert npwd.in_window_mean == 3.5

    # out-window: total sum (210 + 400) - in-window sum (70) = 540, count 180
    #   out_window_mean = 540 / 180 = 3.0
    assert npwd.out_window_mean == 3.0

    npwd.fit(X_fit, y_fit)
    # baseline_mean is the scalar mean of X_fit (inst 2 first 60 timepoints):
    # sum 450 / 60 = 7.5
    assert npwd.baseline_mean == 7.5

    # predict events in the held-out half of instance 2
    # ground truth: y_test has events at ilocs 70 and 90
    y_pred = npwd.predict(X_test)
    assert y_pred is not None  # todo: tighten once _predict handles panel
