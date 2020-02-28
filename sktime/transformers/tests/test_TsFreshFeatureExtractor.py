from sktime.transformers.summarise import TsFreshFeatureExtractor
from sktime.utils.testing import generate_df_from_array
import pandas as pd
import numpy as np
from sktime.datasets import load_gunpoint
from sktime.utils.time_series import time_series_slope
from sktime.pipeline import Pipeline, FeatureUnion
import pytest


# Test output format and dimensions.
@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("len_series", [2, 3])
@pytest.mark.parametrize("n_intervals", [1, 3, 'log', 'sqrt', 'random'])
@pytest.mark.parametrize("features", [[np.mean], [np.mean, np.median],
                                      [np.mean, np.median, np.mean]])
def test_output_format_dim(n_instances, len_series, n_intervals, features):
    X = generate_df_from_array(np.ones(len_series), n_rows=n_instances, n_cols=1)
    n_rows, n_cols = X.shape
    trans = TsFreshFeatureExtractor()
    Xt = trans.transform(X)
    assert isinstance(Xt, pd.DataFrame)
    assert Xt.shape[0] == n_rows

@pytest.mark.parametrize("n_instances", [1, 3])
@pytest.mark.parametrize("len_series", [2, 3])
@pytest.mark.parametrize("n_intervals", [1, 3, 'log', 'sqrt', 'random'])
@pytest.mark.parametrize("features", [[np.mean], [np.mean, np.median],
                                      [np.mean, np.median, np.mean]])
@pytest.mark.parametrize("unsupported_values",[1,-2,(1,3),"bad string",type("hello"),np.ones(5)])                                      
def test_unusupported_rc_parameters(n_instances, len_series, n_intervals, features,unsupported_values):
    with pytest.raises(ValueError):
        X = generate_df_from_array(np.ones(len_series), n_rows=n_instances, n_cols=1)
        n_rows, n_cols = X.shape
        trans = TsFreshFeatureExtractor(default_fc_parameters=unsupported_values)
        Xt = trans.transform(X)


