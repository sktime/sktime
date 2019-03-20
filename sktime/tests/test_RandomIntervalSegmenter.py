from ..transformers.series_to_tabular import RandomIntervalSegmenter
from ..utils.testing import generate_df_from_array
from ..utils.transformations import tabularize
import pytest
import pandas as pd
import numpy as np

N_ITER = 10


# Test output format and dimensions.
def test_output_format_dim():
    for n_cols in [1, 3]:
        for n_rows in [1, 3]:
            for n_obs in [2, 3]:
                for n_intervals in [1, 3, 10, 'sqrt', 'random']:
                    X = generate_df_from_array(np.ones(n_obs), n_rows=n_rows, n_cols=n_cols)

                    trans = RandomIntervalSegmenter(n_intervals=n_intervals)
                    Xt = trans.fit_transform(X)

                    assert isinstance(Xt, (pd.DataFrame, pd.Series))
                    assert Xt.shape[0] == X.shape[0]


# Check that exception is raised for bad input args.
def test_bad_input_args():
    bad_n_intervals = [0, 'abc', 1.0, -1]
    for arg in bad_n_intervals:
        with pytest.raises(ValueError):
            RandomIntervalSegmenter(n_intervals=arg)


# Check if random state always gives same results
def test_random_state():
    X = generate_df_from_array(np.random.normal(size=10))
    random_state = 1234
    trans = RandomIntervalSegmenter(n_intervals='random', random_state=random_state)
    first_Xt = trans.fit_transform(X)
    for _ in range(N_ITER):
        trans = RandomIntervalSegmenter(n_intervals='random', random_state=random_state)
        Xt = trans.fit_transform(X)
        np.testing.assert_array_equal(tabularize(first_Xt).values, tabularize(Xt).values)
