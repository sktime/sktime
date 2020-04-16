import numpy as np
import pandas as pd

from sktime.datasets import load_basic_motions
from sktime.transformers.resizing import TSResizeTransform

# load data
df, _ = load_basic_motions(split='TRAIN', return_X_y=True)


def cut_df_ts(df):
    for row_i in range(df.shape[0]):
        for dim_i in range(df.shape[1]):
            ts = df.at[row_i, f'dim_{dim_i}']
            df.at[row_i, f'dim_{dim_i}'] = pd.Series(ts.tolist()[:len(ts)-dim_i-1])



def test_resizing():
    # cutting the df for each data dim
    ts_lens_before = [len(df.iloc[0][i]) for i in range(len(df.iloc[0]))]
    assert(all([l==ts_lens_before[0] for l in ts_lens_before])) # all equal
    cut_df_ts(df) # inplace operation
    ts_lens_after_cut = [len(df.iloc[0][i]) for i in range(len(df.iloc[0]))]
    assert(not all([l==ts_lens_after_cut[0] for l in ts_lens_after_cut])) # are different

    target_len = 50
    df_resized = TSResizeTransform(target_len).transform(df)
    ts_lens_after_resize = [len(df_resized.iloc[0][i]) for i in range(len(df_resized.iloc[0]))]
    assert(all([l==target_len for l in ts_lens_after_resize]))