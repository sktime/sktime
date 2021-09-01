# -*- coding: utf-8 -*-
import pandas as pd

from sktime.datasets import load_basic_motions
from sktime.transformations.panel.interpolate import TSInterpolator


def cut_X_ts(X):
    for row_i in range(X.shape[0]):
        for dim_i in range(X.shape[1]):
            ts = X.at[row_i, f"dim_{dim_i}"]
            X.at[row_i, f"dim_{dim_i}"] = pd.Series(ts.tolist()[: len(ts) - dim_i - 1])


def test_resizing():
    # 1) all lengths are equal
    # 2) cut lengths and check that they are really different
    # 3) use transformer for resizing to resize time series to equal length
    # 4) check that result length are equal to length that was set for
    #       transformer

    X, _ = load_basic_motions(split="train", return_X_y=True)

    # 1) Check that lengths of all time series (all via the axis=1 - for
    # all dims in first row) are equal.
    ts_lens_before = [len(X.iloc[0][i]) for i in range(len(X.iloc[0]))]
    # all lengths are equal to first length in array
    assert all([length == ts_lens_before[0] for length in ts_lens_before])

    # 2) cutting each time series in each cell of X to make lengths different
    cut_X_ts(X)  # operation is inplace
    # get lengths to ensure that they are really different
    ts_lens_after_cut = [len(X.iloc[0][i]) for i in range(len(X.iloc[0]))]
    assert not all(
        [length == ts_lens_after_cut[0] for length in ts_lens_after_cut]
    )  # are different

    # 3) make tranformer, set target length `target_len` and apply it
    target_len = 50
    Xt = TSInterpolator(target_len).fit_transform(X)

    # 4) check that result time series have lengths equal to `target_len
    #       that we set above
    ts_lens_after_resize = [len(Xt.iloc[0][i]) for i in range(len(Xt.iloc[0]))]
    assert all([length == target_len for length in ts_lens_after_resize])
